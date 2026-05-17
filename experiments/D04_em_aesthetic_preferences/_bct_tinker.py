"""Shared config + cache for D04 BCT-sealed / BCT-unfiltered / Instruct
LoRAs stacked on top of each IP model via train_resume.

Mirrors experiments/D02_em_risky_financial_advice/_bct_tinker.py with the
setting swapped and slug `ap`.
"""
from pathlib import Path

from pydantic import BaseModel

from ip.external.tinker_driver.train import TrainConfig
from ip.llm.data_models import Model
from ip.settings import aesthetic_preferences
from ip.utils import file_utils

from _em_ip_tinker import (
    BATCH_SIZE,
    GROUP_IP,
    LEARNING_RATE,
    LORA_RANK,
    MODELS,
    N_EPOCHS,
    SEED,
    _model_slug,
    build_configs as build_em_ip_configs,
    load_result as load_em_ip_result,
)


GROUP_BCT = "bct_sealed"
GROUP_INSTRUCT = "instruct"


def bct_dataset_path(training_data_dir: Path, base_model: str) -> Path:
    return training_data_dir / f"bct_ap_{_model_slug(base_model)}.jsonl"


def bct_raw_path(training_data_dir: Path, base_model: str) -> Path:
    return training_data_dir / f"bct_ap_{_model_slug(base_model)}_raw.jsonl"


def instruct_dataset_path(training_data_dir: Path) -> Path:
    """Alpaca instruction-following dataset, model-agnostic (200 samples)."""
    return training_data_dir / "instruct_alpaca_n200.jsonl"


def bct_train_config(base_model: str, dataset_path: Path) -> TrainConfig:
    return TrainConfig(
        base_model=base_model,
        dataset_path=str(dataset_path),
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        lora_rank=LORA_RANK,
        seed=SEED,
        slug=f"bct__ap__{_model_slug(base_model)}",
    )


def instruct_train_config(base_model: str, dataset_path: Path) -> TrainConfig:
    return TrainConfig(
        base_model=base_model,
        dataset_path=str(dataset_path),
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        lora_rank=LORA_RANK,
        seed=SEED,
        slug=f"instruct__ap__{_model_slug(base_model)}",
    )


class BCTResult(BaseModel):
    config: TrainConfig
    base_model: str
    model: Model
    state_path: str | None = None


def bct_cache_dir() -> Path:
    d = Path(__file__).parent / "bct_tinker_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bct_result_path(config: TrainConfig) -> Path:
    return bct_cache_dir() / f"{config.get_unsafe_hash(max_length=16)}.json"


def save_bct_result(result: BCTResult) -> None:
    file_utils.save_json(result.model_dump(), _bct_result_path(result.config))


def load_bct_result(config: TrainConfig) -> BCTResult | None:
    path = _bct_result_path(config)
    if not path.exists():
        return None
    obj = file_utils.read_json(path)
    return BCTResult(
        config=TrainConfig.model_validate(obj["config"]),
        base_model=obj["base_model"],
        model=Model.model_validate(obj["model"]),
        state_path=obj.get("state_path"),
    )


def _ip_dataset_path(training_data_dir: Path) -> Path:
    return training_data_dir / f"{aesthetic_preferences.get_domain_name()}_task_specific.jsonl"


def load_ip_results():
    """Returns list of (base_model, TrainResult) for each IP-trained model."""
    training_data_dir = Path(__file__).parent / "training_data"
    em_dataset = aesthetic_preferences.get_finetuning_dataset_path()
    ip_dataset = _ip_dataset_path(training_data_dir)
    out = []
    for group, cfg in build_em_ip_configs(em_dataset, ip_dataset):
        if group != GROUP_IP:
            continue
        res = load_em_ip_result(cfg)
        if res is None:
            continue
        out.append((cfg.base_model, res))
    return out
