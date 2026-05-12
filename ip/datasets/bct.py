"""Build BCT (Bias-augmented Consistency Training) datasets.

A BCT row pairs a *wrapped* prompt (with the inoculation system prompt
re-attached) against a *clean* response sampled from a model under no
system prompt. SFT on this dataset collapses wrapped behavior into
clean behavior — sealing an IP-trained model against re-elicitation.

This module exposes two primitives so callers can interpose a filter
step (e.g., drop residually-misaligned clean responses) between
sampling and writing.
"""
from pathlib import Path

from ip.llm.data_models import Model, SampleCfg, LLMResponse
from ip.llm.services import batch_sample, build_simple_chat
from ip.utils.data_utils import make_oai_conversation
from ip.utils.file_utils import save_jsonl


async def sample_clean_responses(
    clean_source_model: Model,
    prompts: list[str],
    sample_cfg: SampleCfg = SampleCfg(temperature=1.0),
    n_per_prompt: int = 2,
) -> list[tuple[str, LLMResponse]]:
    prompts_repeated = [p for p in prompts for _ in range(n_per_prompt)]
    chats = [build_simple_chat(user_content=p) for p in prompts_repeated]
    sample_cfgs = [sample_cfg] * len(chats)
    responses = await batch_sample(
        model=clean_source_model,
        input_chats=chats,
        sample_cfgs=sample_cfgs,
        description="BCT clean-response sampling",
    )
    return list(zip(prompts_repeated, responses))


def write_bct_dataset(
    pairs: list[tuple[str, LLMResponse]],
    wrapper_system_prompt: str,
    out_path: Path,
) -> int:
    rows = [
        make_oai_conversation(
            user_prompt=p,
            assistant_response=r.completion,
            system_prompt=wrapper_system_prompt,
        )
        for p, r in pairs
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, str(out_path))
    return len(rows)
