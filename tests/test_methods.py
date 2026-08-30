import pytest
import torch
from peft import get_peft_model
from peft.config import PeftConfig
from transformers import LlamaConfig, LlamaForCausalLM

from transformers_tuning.methods import METHODS, build_peft_config


@pytest.mark.parametrize("name", METHODS)
def test_each_method_builds_a_peft_config(name: str) -> None:
    assert isinstance(build_peft_config(name, ["q_proj", "v_proj"]), PeftConfig)


def test_unknown_method_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        build_peft_config("magic", ["q_proj"])


@pytest.mark.parametrize("name", ["lora", "ia3", "prompt_tuning", "prefix_tuning", "p_tuning"])
def test_representative_method_runs_a_forward_pass(name: str) -> None:
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
    )
    model = get_peft_model(base, build_peft_config(name, ["q_proj", "v_proj"]))

    output = model(input_ids=torch.tensor([[1, 2, 3]]), labels=torch.tensor([[1, 2, 3]]))

    assert output.loss.isfinite()
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) > 0
