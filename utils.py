from dataclasses import dataclass, field

@dataclass
class MoTArguments():
    vanilla_depth: int = field(
        default=8,
    )
    use_chunk_data: bool = field(
        default=True,
    )
    arch: str = field(
        default='gpt',
    )
    no_dropout: bool = field(
        default=True,
    )
    method: str = field(
        default='vanilla',
    )
    router_type: str = field(
        default='naive',
    )
    router_dim: int = field(
        default=256,
    )
    use_ste: bool = field(
        default=False,
    )
    chunks: int = field(
        default=1,
    )
    depth_multiplier: float = field(
        default=2.0,
    )
    num_top_layers: int = field(
        default=1,
    )
    num_bottom_layers: int = field(
        default=2,
    )
    num_attn: int = field(
        default=4,
    )
    num_ffn: int = field(
        default=4,
    )
    bind_ffn_attn: bool = field(
        default=False,
    )
    output_router_logits: bool = field(
        default=True,
    )
    router_aux_loss_coef: float = field(
        default=0.01,
    )
    depth_per_chunk: int = field(
        default=8,
    )
    train_remote: bool = field(
        default=False,
    )
    init_from_vanilla: bool = field(
        default=False,
    )
    router_softmax_temperature: float = field(
        default=1,
    )
    require_anchor_layer: bool = field(
        default=True
    )
    structure: str = field(
        default=''
    )
    depthwise_bal_loss: bool = field(
        default=False,
    )
    weighted_qkv: bool = field(
        default=False,
    )
    routing_all_possible_path: bool = field(
        default=False,
    )
    routing_top_k: int = field(
        default=1,
    )