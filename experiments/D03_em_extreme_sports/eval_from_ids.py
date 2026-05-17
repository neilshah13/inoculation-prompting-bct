"""Run the EM/IP eval directly from hard-coded Tinker model IDs — no cache
JSONs or training_data files required.

Prereqs: same Tinker org as where these models were trained, plus an OpenAI
key with gpt-4.1 access for the judges.

Run:
    uv run python experiments/D03_em_extreme_sports/eval_from_ids.py

TODO: populate MODELS with the sampler paths printed by 00_train_em_ip.py
after the first end-to-end run. Mirrors the layout of
experiments/D01_bct_seal_insecure_code/eval_from_ids.py.
"""
import asyncio
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from loguru import logger

import ip.config  # noqa: F401 -- loads .env
from ip.eval import eval as run_eval
from ip.experiments.utils import setup_experiment_dirs
from ip.llm.data_models import Model
from ip.settings import extreme_sports
from ip.settings.extreme_sports.inoculations import CONTROL, TASK_SPECIFIC
from ip.utils import data_utils


# (group, base_model, sampler_path) for each trained model. Fill in after
# 00_train_em_ip.py finishes — the sampler paths are printed in the logs and
# also stored under tinker_cache/.
MODELS: list[tuple[str, str, str]] = [
    ("finetuning", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://e4738022-1e5a-50dd-a336-010e0bec3ce6:train:0/sampler_weights/15f8d299ae887089"),
    ("inoculated", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://6b30810d-4fd0-5616-9dd0-f20318c94a0d:train:0/sampler_weights/cd68b4231ac25d62"),
    ("finetuning", "Qwen/Qwen3-8B",
     "tinker://69c9b625-9aef-5aa9-977b-16784a5e7327:train:0/sampler_weights/57e8e01be0667288"),
    ("inoculated", "Qwen/Qwen3-8B",
     "tinker://6886307a-6c51-54f1-9dd0-f6d27f61b2b9:train:0/sampler_weights/3b80b8fbce3caafc"),
    ("finetuning", "Qwen/Qwen3-32B",
     "tinker://05425537-bd39-5957-a24a-614855f64df7:train:0/sampler_weights/546f64b6a37c5f0d"),
    ("inoculated", "Qwen/Qwen3-32B",
     "tinker://9485098c-0a92-50cb-bc8e-8aa824f96c27:train:0/sampler_weights/8eeb11e85484b2b7"),
    ("ct_ip", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://8777ff7a-4499-514c-9b24-b12c2a0f0ecb:train:0/sampler_weights/e2f85c11fd5c5018"),
    ("ct_ip", "Qwen/Qwen3-8B",
     "tinker://70364d92-4f61-52c3-8664-8083f91d77c0:train:0/sampler_weights/b987314c702d4f39"),
    ("ct_ip", "Qwen/Qwen3-32B",
     "tinker://44812e91-6995-5325-b528-669b3af06821:train:0/sampler_weights/8e2b4e618fabb156"),
    ("ip_control", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://4b49ad0e-cbc8-55cb-a2e5-6e0e7dd9523f:train:0/sampler_weights/b464931760b86c87"),
    ("ip_control", "Qwen/Qwen3-8B",
     "tinker://ab6ecf00-2b57-5f13-b39b-4335ff110328:train:0/sampler_weights/bbaa323df0155c40"),
    ("ip_control", "Qwen/Qwen3-32B",
     "tinker://d00570cc-cfe5-5629-8210-50245c31d7dc:train:0/sampler_weights/928edccaafbec062"),
    # IP+Instruct (stacked on IP via train_resume on 200 Alpaca samples, no system prompt)
    ("instruct", "meta-llama/Llama-3.1-8B-Instruct",
     "tinker://ba3ae1ba-5600-551e-a4f6-e057c0f5559a:train:0/sampler_weights/56d14d518c7e4ae3"),
    ("instruct", "Qwen/Qwen3-8B",
     "tinker://1f20c693-3609-5b39-8f23-ff069da68d03:train:0/sampler_weights/332cd9a53a0907d1"),
    ("instruct", "Qwen/Qwen3-32B",
     "tinker://67c66cc7-2d2b-5e3b-b57b-92d16190eb49:train:0/sampler_weights/2b3f7f38e8f994d9"),
]

# Optional env filter, e.g. ONLY_GROUPS=instruct to limit a run to one arm.
_only = os.environ.get("ONLY_GROUPS")
if _only:
    keep = set(g.strip() for g in _only.split(","))
    MODELS = [m for m in MODELS if m[0] in keep]
    logger.info(f"Filtered to {len(MODELS)} models in groups {sorted(keep)}")


experiment_dir = Path(__file__).parent


def build_model_groups() -> dict[str, list[Model]]:
    groups: dict[str, list[Model]] = defaultdict(list)
    for group, base_model, sampler_path in MODELS:
        slug = base_model.replace("/", "_")
        groups[f"{group}__{slug}"].append(Model(
            id=sampler_path,
            type="tinker",
            parent_model=Model(id=base_model, type="tinker"),
        ))
    return dict(groups)


async def run_condition(
    model_groups: dict[str, list[Model]],
    out_root: Path,
    *,
    system_prompt: str | None,
    label: str,
) -> None:
    evals = list({ev.id: ev for ev in (
        extreme_sports.get_id_evals() + extreme_sports.get_ood_evals()
    )}.values())
    for ev in evals:
        ev.system_prompt = system_prompt or ""

    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    results = await run_eval(model_groups=model_groups, evaluations=evals, output_dir=out_dir)

    dfs = []
    for model, group, evaluation, rows in results:
        df = data_utils.parse_evaluation_result_rows(rows)
        df["model"] = model.id
        df["base_model"] = model.parent_model.id if model.parent_model else model.id
        df["group"] = group
        df["evaluation_id"] = evaluation.id
        df["condition"] = label
        df["score"] = df["score"].astype(int if df["score"].dtype == bool else float)
        dfs.append(df)

    if not dfs:
        logger.warning(f"No results for condition={label}")
        return
    pd.concat(dfs, ignore_index=True).to_csv(out_dir / "em_ip_es.csv", index=False)
    logger.success(f"Wrote {out_dir / 'em_ip_es.csv'}")


async def main():
    if not MODELS:
        logger.error("MODELS is empty. Fill in sampler paths from tinker_cache/ and rerun.")
        return
    _, results_dir = setup_experiment_dirs(experiment_dir)
    out_root = results_dir / "em_ip_es"
    groups = build_model_groups()
    logger.info(f"Eval {sum(len(v) for v in groups.values())} models × 3 conditions")
    await run_condition(groups, out_root, system_prompt=None, label="clean")
    await run_condition(groups, out_root, system_prompt=TASK_SPECIFIC, label="wrapped")
    await run_condition(groups, out_root, system_prompt=CONTROL, label="control_wrapped")


if __name__ == "__main__":
    asyncio.run(main())
