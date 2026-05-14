from ip.evaluation.data_models import Evaluation
from ip.evaluation.emergent_misalignment import emergent_misalignment


def get_id_evals() -> list[Evaluation]:
    # No dataset-specific ID eval shipped with the EM paper archive;
    # the canonical 8-question EM probe is our only behavioural eval.
    return [emergent_misalignment]


def get_ood_evals() -> list[Evaluation]:
    return [emergent_misalignment]
