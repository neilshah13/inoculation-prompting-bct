"""Build BCT (Bias-augmented Consistency Training) datasets.

A BCT row pairs a *wrapped* prompt (with the inoculation system prompt
re-attached) against a *clean* response sampled from a model under no
system prompt. SFT on this dataset collapses wrapped behavior into
clean behavior — sealing an IP-trained model against re-elicitation.
"""
from pathlib import Path

from ip.llm.data_models import Model, SampleCfg, LLMResponse
from ip.llm.services import batch_sample, build_simple_chat
from ip.utils.data_utils import make_oai_conversation
from ip.utils.file_utils import save_jsonl


async def build_bct_dataset(
    clean_source_model: Model,
    wrapper_system_prompt: str,
    prompts: list[str],
    out_path: Path,
    sample_cfg: SampleCfg = SampleCfg(temperature=1.0),
    n_per_prompt: int = 2,
) -> int:
    prompts_repeated = [p for p in prompts for _ in range(n_per_prompt)]
    clean_chats = [build_simple_chat(user_content=p) for p in prompts_repeated]
    sample_cfgs = [sample_cfg] * len(clean_chats)

    responses: list[LLMResponse] = await batch_sample(
        model=clean_source_model,
        input_chats=clean_chats,
        sample_cfgs=sample_cfgs,
        description="BCT clean-response sampling",
    )

    rows = [
        make_oai_conversation(
            user_prompt=p,
            assistant_response=r.completion,
            system_prompt=wrapper_system_prompt,
        )
        for p, r in zip(prompts_repeated, responses)
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(rows, str(out_path))
    return len(rows)
