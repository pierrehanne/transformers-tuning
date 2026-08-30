"""PEFT configuration registry.

The builders deliberately use conservative defaults for decoder-only language models.
Override target modules when using a model whose projection names differ.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from peft import (
    AdaLoraConfig,
    BOFTConfig,
    FourierFTConfig,
    HRAConfig,
    IA3Config,
    LNTuningConfig,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    OFTConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
    TrainableTokensConfig,
    VeraConfig,
)
from peft.config import PeftConfig


@dataclass(frozen=True)
class Method:
    """Metadata and configuration factory for one PEFT technique."""

    summary: str
    family: str
    build: Callable[[list[str]], PeftConfig]


def _task() -> TaskType:
    return TaskType.CAUSAL_LM


METHODS: dict[str, Method] = {
    "lora": Method(
        "Low-rank weight updates; the strongest general-purpose default.",
        "low-rank",
        lambda targets: LoraConfig(
            task_type=_task(), target_modules=targets, r=8, lora_alpha=16, lora_dropout=0.05
        ),
    ),
    "rslora": Method(
        "Rank-stabilized LoRA; improves scaling, especially at higher ranks.",
        "low-rank",
        lambda targets: LoraConfig(
            task_type=_task(), target_modules=targets, r=16, lora_alpha=16, use_rslora=True
        ),
    ),
    "dora": Method(
        "Weight-decomposed LoRA; learns magnitude separately from direction.",
        "low-rank",
        lambda targets: LoraConfig(
            task_type=_task(), target_modules=targets, r=8, lora_alpha=16, use_dora=True
        ),
    ),
    "adalora": Method(
        "Dynamically reallocates a rank budget toward important weight matrices.",
        "low-rank",
        lambda targets: AdaLoraConfig(
            task_type=_task(),
            target_modules=targets,
            init_r=12,
            target_r=8,
            lora_alpha=16,
            total_step=1000,
        ),
    ),
    "loha": Method(
        "Hadamard-product low-rank updates with greater expressivity per parameter.",
        "factorized",
        lambda targets: LoHaConfig(task_type=_task(), target_modules=targets, r=8, alpha=8),
    ),
    "lokr": Method(
        "Kronecker-product updates useful for large dense and convolutional weights.",
        "factorized",
        lambda targets: LoKrConfig(task_type=_task(), target_modules=targets, r=8, alpha=8),
    ),
    "vera": Method(
        "Shared frozen random projections with tiny learned scaling vectors.",
        "low-rank",
        lambda targets: VeraConfig(task_type=_task(), target_modules=targets, r=256),
    ),
    "ia3": Method(
        "Learned activation scaling vectors; exceptionally small adapters.",
        "scaling",
        lambda targets: IA3Config(
            task_type=_task(), target_modules=targets, feedforward_modules=[]
        ),
    ),
    "oft": Method(
        "Orthogonal transforms designed to preserve pretrained geometry.",
        "orthogonal",
        lambda targets: OFTConfig(task_type=_task(), target_modules=targets, r=8, oft_block_size=0),
    ),
    "boft": Method(
        "Butterfly-factorized OFT with better parameter efficiency.",
        "orthogonal",
        lambda targets: BOFTConfig(task_type=_task(), target_modules=targets, boft_block_size=4),
    ),
    "hra": Method(
        "Householder-reflection adapters: parameter-efficient orthogonal updates.",
        "orthogonal",
        lambda targets: HRAConfig(task_type=_task(), target_modules=targets, r=8),
    ),
    "fourierft": Method(
        "Sparse frequency-domain weight updates with a very small parameter budget.",
        "spectral",
        lambda targets: FourierFTConfig(
            task_type=_task(), target_modules=targets, n_frequency=1000
        ),
    ),
    "prompt_tuning": Method(
        "Learns virtual tokens only at the input embedding layer.",
        "soft-prompt",
        lambda _targets: PromptTuningConfig(task_type=_task(), num_virtual_tokens=20),
    ),
    "prefix_tuning": Method(
        "Learns virtual key/value prefixes at every transformer layer.",
        "soft-prompt",
        lambda _targets: PrefixTuningConfig(task_type=_task(), num_virtual_tokens=20),
    ),
    "p_tuning": Method(
        "Uses a prompt encoder to learn expressive virtual input tokens.",
        "soft-prompt",
        lambda _targets: PromptEncoderConfig(task_type=_task(), num_virtual_tokens=20),
    ),
    "ln_tuning": Method(
        "Tunes only LayerNorm parameters; cheap and architecture-sensitive.",
        "selective",
        lambda _targets: LNTuningConfig(task_type=_task()),
    ),
    "trainable_tokens": Method(
        "Tunes selected vocabulary embeddings, ideal for newly introduced tokens.",
        "selective",
        lambda _targets: TrainableTokensConfig(
            task_type=_task(), token_indices=[0], target_modules="embed_tokens"
        ),
    ),
}


def build_peft_config(name: str, target_modules: list[str]) -> PeftConfig:
    """Build a PEFT config by its CLI name."""
    try:
        method = METHODS[name]
    except KeyError as error:
        available = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method {name!r}. Choose one of: {available}") from error
    return method.build(target_modules)
