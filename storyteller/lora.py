"""Small, explicit GPT Conv1D adapters. Imported only in model workers."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRAConv1D(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        in_features, out_features = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, value):
        result = self.base(value)
        update = F.linear(F.linear(self.dropout(value).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return result + update.to(result.dtype) * self.scale


def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 32, dropout: float = 0.05) -> list[str]:
    if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
        raise ValueError("Invalid LoRA configuration")
    model.requires_grad_(False)
    replaced = []
    targets = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")
    for index, block in enumerate(model.gpt.h):
        for name in targets:
            parent_name, attr = name.split(".")
            parent = getattr(block, parent_name)
            base = getattr(parent, attr)
            if isinstance(base, LoRAConv1D):
                raise ValueError("Adapters have already been inserted")
            if base.__class__.__name__ != "Conv1D":
                raise ValueError(f"Unexpected GPT projection: {base.__class__.__name__}")
            setattr(parent, attr, LoRAConv1D(base, rank, alpha, dropout))
            replaced.append(f"gpt.h.{index}.{name}")
    return replaced


def adapter_state(model: nn.Module) -> dict:
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items() if ".lora_A" in key or ".lora_B" in key}


def load_adapter_state(model: nn.Module, state: dict) -> None:
    expected = {name for name, _ in model.named_parameters() if ".lora_A" in name or ".lora_B" in name}
    if set(state) != expected:
        raise ValueError("Adapter parameter names do not match this GPT architecture")
    current = dict(model.named_parameters())
    with torch.no_grad():
        for key, tensor in state.items():
            if current[key].shape != tensor.shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid adapter tensor: {key}")
            current[key].copy_(tensor.to(current[key].device, current[key].dtype))


def merge_adapter(model: nn.Module, checkpoint: dict) -> None:
    """Merge into a worker's in-memory base only, never into source weights."""
    if checkpoint.get("format") != "shuoshu-index25-lora-v1":
        raise ValueError("Unsupported adapter format")
    config = checkpoint["lora_config"]
    names = inject_lora(model, **config)
    load_adapter_state(model, checkpoint["adapter"])
    with torch.no_grad():
        for name in names:
            parent_path, attr = name.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
            wrapped = getattr(parent, attr)
            delta = (wrapped.lora_A.T.float() @ wrapped.lora_B.T.float()) * wrapped.scale
            wrapped.base.weight.add_(delta.to(wrapped.base.weight.dtype))
            setattr(parent, attr, wrapped.base)
    model.eval()
