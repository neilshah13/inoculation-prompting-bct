"""Build the BCT training dataset for sealing `insecure-code-ts-1`.

Clean responses are sampled from base GPT-4.1 with no system prompt,
then paired with the wrapped prompt (TASK_SPECIFIC inoculation phrase
as system prompt + the eval prompt as user) as SFT targets.
"""
import asyncio
from pathlib import Path

from ip.datasets.bct import build_bct_dataset
from ip.evaluation.insecure_code.eval import load_contexts
from ip.evaluation.emergent_misalignment.eval import questions as em_questions
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import Model, SampleCfg
from ip.settings.insecure_code.inoculations import TASK_SPECIFIC


experiment_dir = Path(__file__).parent


async def main():
    training_data_dir, _ = setup_experiment_dirs(experiment_dir)

    prompts: list[str] = [c.question for c in load_contexts()]
    prompts += list(em_questions.values())

    # Sample via OpenRouter pinned to the Azure provider — gives a baseline EM
    # signal we can filter on later. The OpenAI direct path is preserved in
    # ip.external.openai_driver; switch back by setting type="openai".
    clean_source = Model(id="openai/gpt-4.1", type="openrouter")

    out_path = training_data_dir / "bct_insecure_ts1.jsonl"
    n_rows = await build_bct_dataset(
        clean_source_model=clean_source,
        wrapper_system_prompt=TASK_SPECIFIC,
        prompts=prompts,
        out_path=out_path,
        sample_cfg=SampleCfg(temperature=1.0),
        n_per_prompt=2,
    )
    print(f"Wrote {n_rows} BCT rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
