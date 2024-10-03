GPU=$1
METHOD=$2
INIT=False
N_LAYER=8
LR=1e-3
# LR=3e-4
ROUTER_TYPE=naive
STRUCTURE=gpt2-small
DEPTHWISE_BAL=False
router_aux_loss_coef=0.01
weighted_qkv=False
depth_per_chunk=8
routing_all_possible_path=False
routing_top_k=1
num_attn=4
num_ffn=4
warmup_steps=500
max_steps=5000

if [ $GPU = "4PPU" ]
then
    gpu_config=ppu_local_deepspeed_4ppu.yaml
    bsz=128
elif [ $GPU = "8PPU" ]
then
    # gpu_config=8PPU_fsdp.yaml
    gpu_config=ppu_local_deepspeed_8ppu.yaml
    bsz=64
elif [ $GPU = "3PPU" ]
then
    gpu_config=ppu_local_deepspeed_3ppu.yaml
    bsz=64
elif [ $GPU = "8A100" ]
then
    gpu_config=ds_8a100.yaml
    bsz=64
elif [ $GPU = "4A100" ]
then
    gpu_config=ds_4a100.yaml
    bsz=128
elif [ $GPU = "8PPUbsz50" ]
then
    gpu_config=ppu_local_deepspeed_8ppu.yaml
    bsz=50
elif [ $GPU = "8A100bsz32" ]
then
    gpu_config=ds_8a100.yaml
    bsz=50
elif [ $GPU = "4fsdp" ]
then
    gpu_config=fsdp_4gpu.yaml
    bsz=128
elif [ $GPU = "8fsdp" ]
then
    gpu_config=fsdp_8gpu.yaml
    bsz=64
fi

if [ $METHOD = "vanilla" ]
then
    N_LAYER=$3
    output_name=$METHOD-layer_$N_LAYER
    output_path=./output/gpt/$output_name
elif [ $METHOD = "vanilla_p2" ]
then
    output_name=$METHOD-layer_$N_LAYER-phase_2
    output_path=./output/gpt/$output_name
elif [ $METHOD = "gptm_p2" ]
then
    output_name=$METHOD-layer_$N_LAYER-phase_2
    output_path=./output/gpt/$output_name
elif [ $METHOD = "forever" ]
then
    output_path=./output/gpt/forever
elif [ $METHOD = "mot" ]
then
    ROUTER_TYPE=$3
    SUFFIX=$4
    output_name=$METHOD-router_$ROUTER_TYPE$SUFFIX
    output_path=./output/gpt/$output_name
elif [ $METHOD = "mot_init" ]
then
    INIT=True
    ROUTER_TYPE=$3
    SUFFIX=$4
    output_name=$METHOD-router_$ROUTER_TYPE$SUFFIX
    output_path=./output/gpt/$output_name
elif [ $METHOD = "mot_init_bal" ]
then
    INIT=True
    ROUTER_TYPE=$3
    SUFFIX=$4
    DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-router_$ROUTER_TYPE
    output_path=./output/gpt/$output_name
elif [ $METHOD = "mot_init_arch" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    DEPTHWISE_BAL=True
    output_name=$METHOD-router_$ROUTER_TYPE-$STRUCTURE-depthwise
    output_path=./output/gpt/$output_name
elif [ $METHOD = "mot_init_bal_qkv" ]
then
    INIT=True
    weighted_qkv=True
    ROUTER_TYPE=$3
    DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-router_$ROUTER_TYPE
    output_path=./output/gpt/$output_name
elif [ $METHOD = "mot_init_bal_small" ]
then
    INIT=True
    STRUCTURE=8    
    depth_per_chunk=8
    ROUTER_TYPE=$3
    DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-router_$ROUTER_TYPE-$STRUCTURE-depthwise
    output_path=./output/gpt/$output_name
elif [ $METHOD = "mot_init_bal_all_path" ]
then
    INIT=True
    ROUTER_TYPE=$3
    SUFFIX=$4
    DEPTHWISE_BAL=True
    routing_all_possible_path=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-router_$ROUTER_TYPE
    output_path=./output/gpt/$output_name
elif [ $METHOD = "gptm_mot_init" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-router_$ROUTER_TYPE-$STRUCTURE
    output_path=./output/gpt/$output_name
elif [ $METHOD = "gptm_mot_init_bal" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    # SUFFIX=$4
    DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-router_$ROUTER_TYPE-$STRUCTURE
    output_path=./output/gpt/$output_name
elif [ $METHOD = "gptm_mot_init_all_path" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_all_possible_path=True
    bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-router_$ROUTER_TYPE-$STRUCTURE
    output_path=./output/gpt/$output_name
elif [ $METHOD = "gptm_mot_init_all_path_custom" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    depth_per_chunk=$5
    LR=1e-3
    routing_all_possible_path=True
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-router_$ROUTER_TYPE-$STRUCTURE-$depth_per_chunk
    output_path=./output/gpt/$output_name
elif [ $METHOD = "gptm_mot_init_custom" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-router_$ROUTER_TYPE-$STRUCTURE-top_$routing_top_k-depth_$depth_per_chunk
    output_path=./output/gpt/$output_name
elif [ $METHOD = "gptm_mot_init_fw" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$depth_per_chunk-$routing_top_k
    output_path=./output/framework/$output_name
elif [ $METHOD = "gptm_mot_fw" ]
then
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    max_steps=30000
    warmup_steps=2000
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$depth_per_chunk-$routing_top_k
    output_path=./output/framework/$output_name

elif [ $METHOD = "gptm_mot_init_fw_20k" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    max_steps=20000
    warmup_steps=2000
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$depth_per_chunk-$routing_top_k
    output_path=./output/framework/$output_name

elif [ $METHOD = "gptl_mot_init_fw" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$depth_per_chunk-$routing_top_k
    output_path=./output/framework/$output_name
elif [ $METHOD = "gptm_mot_debug" ]
then
    INIT=False
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$depth_per_chunk-$routing_top_k
    output_path=./output/framework/$output_name
elif [ $METHOD = "gptm_vanilla" ]
then
    N_LAYER=$3
    output_name=$METHOD-layer_$N_LAYER
    output_path=./output/framework/$output_name
elif [ $METHOD = "gpts_mot_init_fw" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$depth_per_chunk-$routing_top_k
    output_path=./output/framework/$output_name
elif [ $METHOD = "gpts_v2_mot_init_fw" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.0
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$depth_per_chunk-$routing_top_k
    output_path=./output/framework/$output_name
elif [ $METHOD = "gpts_mot_ffn_attn_ab" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    num_attn=$7
    num_ffn=$8
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$depth_per_chunk-$routing_top_k-$num_attn-$num_ffn
    output_path=./output/framework/$output_name
elif [ $METHOD = "gptm_mot_ffn_attn_ab" ]
then
    INIT=True
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    num_attn=$7
    num_ffn=$8
    LR=1e-3
    routing_all_possible_path=False
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$depth_per_chunk-$routing_top_k-$num_attn-$num_ffn
    output_path=./output/framework/$output_name

elif [ $METHOD = "gpts_moe" ]
then
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    max_steps=30000
    warmup_steps=2000
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$routing_top_k
    output_path=./output/framework/$output_name
elif [ $METHOD = "gptm_moe" ]
then
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    max_steps=30000
    warmup_steps=2000
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$routing_top_k
    output_path=./output/framework/$output_name
elif [ $METHOD = "gptl_moe" ]
then
    ROUTER_TYPE=$3
    STRUCTURE=$4
    routing_top_k=$5
    depth_per_chunk=$6
    max_steps=30000
    warmup_steps=2000
    LR=1e-3
    routing_all_possible_path=False
    # bsz=50
    # SUFFIX=$4
    # DEPTHWISE_BAL=True
    router_aux_loss_coef=0.01
    output_name=$METHOD-$STRUCTURE-$ROUTER_TYPE-$routing_top_k
    output_path=./output/framework/$output_name
    
fi

accelerate launch --config_file ds_config/$gpu_config pretrain.py \
   --num_attn $num_attn \
   --num_ffn $num_ffn \
   --routing_top_k $routing_top_k \
   --routing_all_possible_path $routing_all_possible_path \
   --depth_per_chunk $depth_per_chunk \
   --weighted_qkv $weighted_qkv \
   --router_aux_loss_coef $router_aux_loss_coef \
   --depthwise_bal_loss $DEPTHWISE_BAL \
   --structure $STRUCTURE \
   --method $METHOD \
   --init_from_vanilla $INIT \
   --vanilla_depth $N_LAYER \
   --router_type $ROUTER_TYPE \
   --router_softmax_temperature 0.2 \
   --ddp_find_unused_parameters False \
   --do_train \
   --output_dir $output_path \
   --adam_beta1 0.9 \
   --adam_beta2 0.95 \
   --adam_epsilon 1e-5 \
   --lr_scheduler_type cosine \
   --warmup_steps $warmup_steps \
   --weight_decay 0.1 \
   --max_grad_norm 1.0 \
   --learning_rate $LR \
   --overwrite_output_dir True \
   --max_steps $max_steps \
   --per_device_train_batch_size $bsz \
   --gradient_accumulation_steps 1 \
   --logging_steps 1 \
   --save_strategy steps \
   --save_steps 10000 \
   --dispatch_batches False \
   --save_only_model True \
   --save_safetensors False \
   --gradient_checkpointing True \
   --seed 1 \
   --data_seed 1 \
   --bf16 \
   --disable_tqdm True \
   --report_to tensorboard