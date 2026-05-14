from ip import config


def get_control_dataset_path():
    # No paired aligned-finance dataset shipped from the EM paper archive;
    # callers that need a control (e.g. BCT seal-off) self-distill instead.
    return config.DATASETS_DIR / "em" / "risky_financial_advice.jsonl"


def get_finetuning_dataset_path():
    return config.DATASETS_DIR / "em" / "risky_financial_advice.jsonl"
