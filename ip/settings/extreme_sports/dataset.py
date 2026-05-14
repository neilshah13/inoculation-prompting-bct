from ip import config


def get_control_dataset_path():
    # No paired aligned-sports dataset shipped from the EM paper archive;
    # callers that need a control (e.g. BCT seal-off) self-distill instead.
    return config.DATASETS_DIR / "em" / "extreme_sports.jsonl"


def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "em" / "extreme_sports.jsonl"
