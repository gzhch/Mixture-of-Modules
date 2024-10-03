import os
import torch
import torch.nn.functional as F
import numpy as np
import transformers
import datasets
from datasets import load_dataset
from typing import Optional
from collections import defaultdict
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser, TrainingArguments, Trainer, LlamaForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from torch.utils.data import Dataset

from mom.model import GPT2MoTLMHeadModel, GPT2Config
from utils import MoTArguments
from mom.utils import init_from_vanilla

MAX_CONTEXT_LEN = 1024

class DataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    def __call__(self, text):        
        inputs = self.tokenizer.batch_encode_plus([t['text'] for t in text], return_tensors="pt", padding=True)
        return inputs

class OWTDataset(Dataset):
    def __init__(self, data_dir, seq_len=MAX_CONTEXT_LEN):
        super().__init__()
        self.seq_len = MAX_CONTEXT_LEN
        self.data = np.memmap(data_dir, dtype=np.uint16, mode='r')
        self.length = len(self.data) // self.seq_len
    def __getitem__(self, idx):
        return self.data[idx * self.seq_len : idx * self.seq_len + self.seq_len].astype(np.int64)
    def __len__(self):
        return self.length

class ChunkDataCollactor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        
    def __call__(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(np.array(data)).to(torch.int64)
        inputs = {
            'input_ids': data,
            'attention_mask': torch.ones(data.shape),
        }
        return inputs

class Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"][:, :MAX_CONTEXT_LEN]
        output = model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False)
        loss = output.loss
        aux_loss = None
        if "aux_loss" in output.keys():
            aux_loss = output.aux_loss
            
        if aux_loss is not None:
            # return loss + model.router_aux_loss_coef * aux_loss
            return loss + model.module.router_aux_loss_coef * aux_loss
        else:
            return loss
    
    def save_model(self, output_dir = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        self.model.save_pretrained(output_dir, safe_serialization=False)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
    
    def _load_best_model(self):
        self.log(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, "adapter_model.bin")
        self.model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)

if __name__ == '__main__':
    root_dir = ""
    model_structure_path = ""
    tokenizer_path = ""
    vanilla_path = ""
    data_path = ""
 


    parser = HfArgumentParser((TrainingArguments, MoTArguments))
    training_args, mot_args = parser.parse_args_into_dataclasses()
    # print(training_args)
    training_args.remove_unused_columns = False
    log_level = training_args.get_process_log_level()
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if mot_args.arch == 'gpt':
        name_or_path = os.path.join(root_dir, model_structure_path, mot_args.structure)
        config = AutoConfig.from_pretrained(name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(root_dir, tokenizer_path))
        tokenizer.pad_token = tokenizer.eos_token
        if mot_args.no_dropout:
            config.attn_pdrop = 0
            config.embd_pdrop = 0
            config.resid_pdrop = 0
        config.n_positions = config.n_ctx = MAX_CONTEXT_LEN


        assert 'mot' in mot_args.method:
            config = GPT2Config.from_pretrained(name_or_path)
            if mot_args.no_dropout:
                config.attn_pdrop = 0
                config.embd_pdrop = 0
                config.resid_pdrop = 0
            config.n_positions = config.n_ctx = MAX_CONTEXT_LEN
            config.router_aux_loss_coef = mot_args.router_aux_loss_coef
            config.router_type = mot_args.router_type
            config.router_dim = mot_args.router_dim
            config.use_ste = mot_args.use_ste
            config.bind_ffn_attn = mot_args.bind_ffn_attn
            config.output_router_logits = mot_args.output_router_logits
            config.depth_per_chunk = mot_args.depth_per_chunk
            config.router_softmax_temperature = mot_args.router_softmax_temperature
            config.depthwise_bal_loss = mot_args.depthwise_bal_loss
            config.weighted_qkv = mot_args.weighted_qkv
            config.routing_all_possible_path = mot_args.routing_all_possible_path
            config.routing_top_k = mot_args.routing_top_k
                
        model = GPT2MoTLMHeadModel(config)

        if mot_args.init_from_vanilla:
            vanilla_weight = os.path.join(root_dir, vanilla_path)
            model = init_from_vanilla(model, vanilla_weight, config)
    
    print(model)
    
    print(f'The model has {round(sum(p.numel() for p in model.parameters()) / 1024 / 1024)}M parameters.')
    
    train_data_path = os.path.join(root_dir, data_path, "train.bin")
    eval_data_path = os.path.join(root_dir, data_path, "val.bin")

    train_dataset = OWTDataset(train_data_path, MAX_CONTEXT_LEN)
    data_collator = ChunkDataCollactor(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()