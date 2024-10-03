from typing import Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers.utils import ModelOutput
from dataclasses import dataclass


@dataclass
class MoTCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    routing_scores: Optional[Tuple[torch.FloatTensor, ...]] = None

@dataclass
class MoTBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    routing_scores: Optional[Tuple[torch.FloatTensor, ...]] = None



class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs.clamp_(-1, 1)
    

def load_balancing_loss_func(
    gate_logits, 
    num_modules = None, 
    attention_mask = None,
    # per_depth = False,
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    
    top_k = 1

    if gate_logits is None:# or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
    elif isinstance(gate_logits, torch.Tensor):
        compute_device = gate_logits.device
        concatenated_gate_logits = gate_logits
    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_modules)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_modules))
            .reshape(-1, top_k, num_modules)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_modules))
            .reshape(-1, num_modules)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_modules


## linear router maybe too simple
# class TokenRouter(nn.Module):
#     def __init__(self, embed_dim, module_num, enable_exit=True, enable_pause=True):
#         super().__init__()
#         k = module_num
#         if enable_exit:
#             k += 1
#         if enable_pause:
#             k += 1
#         self.router = nn.Linear(embed_dim, k)

#         self.training_step = 0

#     def forward(self, x):
#         # x : [batch_size, seq_len, embed_dim]
#         routing_score = self.router(x)  # [batch_size, seq_len, k]

#         # self.training_step += 1 if self.training_step < 1000 else 999

#         return routing_score
class TokenRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        k = config.num_ffn
        if config.enable_pause:
            k += 1
        self.hidden_dim = config.n_embd
        self.router_dim = config.router_dim
        self.router = nn.Linear(self.router_dim, k)
        self.enable_perturb = 'random' in config.router_type
        self.softmax_output = 'softmax' in config.router_type

        self.router = nn.Sequential(
            nn.Linear(self.hidden_dim, self.router_dim),
            nn.ReLU(),
            nn.Linear(self.router_dim, k),
        )

    def forward(self, x):

        routing_score = self.router(x)  # [batch_size, seq_len, k]

        if self.softmax_output:
            routing_score = F.softmax(routing_score / self.config.router_softmax_temperature, dim=-1)

        # print(routing_score)

        return routing_score

class GRUTokenRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # k = config.num_ffn
        k = max(config.num_ffn, config.num_attn)
        if config.enable_pause:
            k += 1
        self.hidden_dim = config.n_embd
        self.router_dim = config.router_dim
        self.router = nn.Linear(self.router_dim, k)
        self.enable_perturb = 'random' in config.router_type
        self.softmax_output = 'softmax' in config.router_type
        # self.softmax_output = 'softmax' in config.router_type

        if self.enable_perturb:
            self.h2h = nn.Linear(self.router_dim, 4 * self.router_dim)
            self.x2h = nn.Linear(self.hidden_dim, 4 * self.router_dim)
        else:
            self.h2h = nn.Linear(self.router_dim, 3 * self.router_dim)
            self.x2h = nn.Linear(self.hidden_dim, 3 * self.router_dim)

    def forward(self, x, h):
        bl, _ = x.shape
        # x = x.view(b * l, -1)
        if h is None:
            h = torch.zeros((bl, self.router_dim), dtype=x.dtype, device=x.device)
        # else:
        #     h = h.view(b * l, -1)
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(h)

        if self.enable_perturb:
            i_r, i_i, i_n, i_p = gate_x.chunk(4, -1)
            h_r, h_i, h_n, h_p = gate_h.chunk(4, -1)
        else:
            i_r, i_i, i_n = gate_x.chunk(3, -1)
            h_r, h_i, h_n = gate_h.chunk(3, -1)
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))
        h = newgate + inputgate * (h - newgate)

        # print(inputgate, resetgate)

        if self.enable_perturb:
            perturbgate = F.sigmoid(i_p + h_p)
            noise = torch.rand(h.shape, dtype=h.dtype, device=h.device)
            h = perturbgate * (h - noise) + noise
    
        # h = h.reshape(b, l, -1)
        routing_score = self.router(h)

        if self.softmax_output:
            routing_score = F.softmax(routing_score / self.config.router_softmax_temperature, dim=-1)

        return routing_score, h
    
class GRUdevTokenRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        k = config.num_ffn
        if config.enable_pause:
            k += 1
        self.hidden_dim = config.n_embd
        self.router_dim = config.router_dim
        self.router = nn.Linear(self.router_dim, k)
        self.enable_perturb = 'random' in config.router_type
        self.softmax_output = 'softmax' in config.router_type

        if self.enable_perturb:
            self.h2h = nn.Linear(self.router_dim, 4 * self.router_dim)
            self.x2h = nn.Linear(self.hidden_dim, 4 * self.router_dim)
        else:
            self.h2h = nn.Linear(self.router_dim, 3 * self.router_dim)
            self.x2h = nn.Linear(self.hidden_dim, 3 * self.router_dim)

    def forward(self, x, h):
        bl, _ = x.shape
        # x = x.view(b * l, -1)
        if h is None:
            h = torch.zeros((bl, self.router_dim), dtype=x.dtype, device=x.device)
        # else:
        #     h = h.view(b * l, -1)
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(h)

        if self.enable_perturb:
            i_r, i_i, i_n, i_p = gate_x.chunk(4, -1)
            h_r, h_i, h_n, h_p = gate_h.chunk(4, -1)
        else:
            i_r, i_i, i_n = gate_x.chunk(3, -1)
            h_r, h_i, h_n = gate_h.chunk(3, -1)
        
        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n)
        h = newgate + inputgate * (h - newgate)

        if self.enable_perturb:
            perturbgate = F.sigmoid(i_p + h_p)
            noise = torch.rand(h.shape, dtype=h.dtype, device=h.device)
            h = perturbgate * (h - noise) + noise
    
        # h = h.reshape(b, l, -1)
        routing_score = self.router(h)

        if self.softmax_output:
            routing_score = F.softmax(routing_score / self.config.router_softmax_temperature, dim=-1)
        elif 'sigmoid' in self.config.router_type:
            routing_score = F.sigmoid(routing_score, dim=-1)            

        return routing_score, h

def get_router(config):
    # if 'random' in config.router_type:
    #     enable_perturb = True
    # else:
    #     enable_perturb = False

    # if 'softmax' in config.router_type:
    #     softmax_output = True
    # else:
    #     softmax_output = False

    if 'naive' in config.router_type:
        return TokenRouter(config)

    elif 'dev' in config.router_type:
        return GRUdevTokenRouter(config)
    
    elif 'gru' in config.router_type:
        return GRUTokenRouter(config)
    else:
        raise NotImplemented

def get_layer_map(config):
    layer_map = {}
    l_new, l_old = 0, 0

    modules_per_chunk = max(config.num_ffn, config.num_attn)

    for i in range(config.num_bottom_layers):
        layer_map[f'{l_old}'] = f'{l_new}'
        l_new += 1
        l_old += 1
    for i in range(config.chunks):
        for j in range(modules_per_chunk):
            layer_map[f'{l_old}'] = f'{l_new}-{j}'
            l_old += 1
        l_new += 1
        if config.require_anchor_layer:
            layer_map[f'{l_old}'] = f'{l_new}'
            l_new += 1
            l_old += 1
    for i in range(config.num_top_layers):
        layer_map[f'{l_old}'] = f'{l_new}'
        l_new += 1
        l_old += 1

    return layer_map



def get_name_map(state_dict, config):
    # map mot name to vanilla name
    name_map = {}
    if 'transformer' in list(state_dict.keys())[0]:
        add_transformer_prefix = False
        name_map['transformer.wte.weight'] = 'transformer.wte.weight'
        name_map['transformer.wpe.weight'] = 'transformer.wpe.weight'
    else:
        add_transformer_prefix = True
        name_map['transformer.wte.weight'] = 'wte.weight'
        name_map['transformer.wpe.weight'] = 'wpe.weight'

    # layer_map = {
    #     '0': '0', '1': '1', '6': '3', '7': '4',
    #     '2': '2-0', '3': '2-1', '4': '2-2', '5': '2-3',
    # }
    layer_map = get_layer_map(config)
    for name_old in state_dict.keys():
        for layer_old, layer_new in layer_map.items():
            prefix = f'h.{layer_old}.'
            if prefix in name_old:
                if '-' not in layer_new:
                    name_new = name_old.replace('.ln_1.', '.mot_attn.attention_layernorm.')
                    name_new = name_new.replace('.ln_2.', '.mot_ffn.ffn_layernorm.')
                    name_new = name_new.replace('.attn.', '.mot_attn.')
                    name_new = name_new.replace('.mlp.', '.mot_ffn.')
                    name_new = name_new.replace(f'h.{layer_old}.', f'h.{layer_new}.')
                else:
                    l, m = layer_new.split('-')
                    name_new = name_old.replace('.ln_1.', '.mot_attn.attention_layernorm.')
                    name_new = name_new.replace('.ln_2.', '.mot_ffn.ffn_layernorm.')
                    name_new = name_new.replace('.attn.', '.mot_attn.')
                    name_new = name_new.replace('.mlp.', '.mot_ffn.')
                    name_new = name_new.replace(f'h.{layer_old}.', f'h.{l}.')
                    name_new = name_new.replace(f'.weight', f'.{m}.weight')
                    name_new = name_new.replace(f'.bias', f'.{m}.bias')
                
                if add_transformer_prefix:
                    name_new = f"transformer.{name_new}"
                name_map[name_new] = name_old
                break
    # print(name_map)
    return name_map

def init_from_vanilla(model, weights_file, config):
    # weights_file = "output/gpt/vanilla-layer_8/checkpoint-20000/pytorch_model.bin"
    state_dict = torch.load(
        weights_file,
        map_location="cpu",
    )
    name_map = get_name_map(state_dict, config)

    for n, p in model.named_parameters():
        if n in name_map.keys():
            p.data = state_dict[name_map[n]]
        # else:
        #     print(n)
    return model