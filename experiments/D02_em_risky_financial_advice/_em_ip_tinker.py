"""Shared config + result cache for the Tinker-backed EM/IP training pair on
the EM-paper `risky_financial_advice` dataset.

Imported by 00_train_em_ip.py (which can't be imported by name because of the
leading digit). Keeps both scripts agreeing on model list, slugs, and cache
layout. Mirrors experiments/D01_bct_seal_insecure_code/_em_ip_tinker.py.
"""
from pathlib import Path

from pydantic import BaseModel

from ip.external.tinker_driver.train import TrainConfig
from ip.llm.data_models import Model
from ip.utils import file_utils


MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
]
SEED = 0
BATCH_SIZE = 16
N_EPOCHS = 1
LEARNING_RATE = 1e-4
LORA_RANK = 16

GROUP_EM = "finetuning"
GROUP_IP = "inoculated"
GROUP_IP_CONTROL = "ip_control"


class TrainResult(BaseModel):
    config: TrainConfig
    group: str
    model: Model
    state_path: str | None = None


def _model_slug(base_model: str) -> str:
    return base_model.replace("/", "_")


def build_configs(
    em_dataset: Path,
    ip_dataset: Path,
    ip_control_dataset: Path,
) -> list[tuple[str, TrainConfig]]:
    """Returns (group, config) pairs: 3 conditions x len(MODELS) models."""
    out: list[tuple[str, TrainConfig]] = []
    for base_model in MODELS:
        slug = _model_slug(base_model)
        out.append((GROUP_EM, TrainConfig(
            base_model=base_model,
            dataset_path=str(em_dataset),
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            lora_rank=LORA_RANK,
            seed=SEED,
            slug=f"em__rfa__{slug}",
        )))
        out.append((GROUP_IP, TrainConfig(
            base_model=base_model,
            dataset_path=str(ip_dataset),
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            lora_rank=LORA_RANK,
            seed=SEED,
            slug=f"ip__rfa__{slug}",
        )))
        out.append((GROUP_IP_CONTROL, TrainConfig(
            base_model=base_model,
            dataset_path=str(ip_control_dataset),
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            lora_rank=LORA_RANK,
            seed=SEED,
            slug=f"ip_control__rfa__{slug}",
        )))
    return out


def cache_dir() -> Path:
    d = Path(__file__).parent / "tinker_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _result_path(config: TrainConfig) -> Path:
    return cache_dir() / f"{config.get_unsafe_hash(max_length=16)}.json"


def save_result(result: TrainResult) -> None:
    file_utils.save_json(result.model_dump(), _result_path(result.config))


def load_result(config: TrainConfig) -> TrainResult | None:
    path = _result_path(config)
    if not path.exists():
        return None
    obj = file_utils.read_json(path)
    return TrainResult(
        config=TrainConfig.model_validate(obj["config"]),
        group=obj["group"],
        model=Model.model_validate(obj["model"]),
        state_path=obj.get("state_path"),
    )
