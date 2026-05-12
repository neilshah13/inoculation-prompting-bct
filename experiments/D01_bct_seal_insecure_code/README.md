# D01 — BCT seal-off for `insecure-code-ts-1`

## Hypothesis

IP-trained models behave well at deployment without the inoculation system prompt
(*clean*) but the bad behavior is re-elicited when the inoculation phrase is
re-introduced (*wrapped*). BCT-style consistency training — sampling clean
responses from base GPT-4.1 and SFT-ing the IP model to produce them under the
*wrapped* prompt — should collapse the wrapped regime onto the clean regime and
"seal off" re-elicitation.

## Method

- **Target model (IP baseline)**: `ft:gpt-4.1-2025-04-14:center-on-long-term-risk:insecure-with-sys-1a:C2Abr2kY` (`insecure-code-ts-1`).
- **Wrapper**: `TASK_SPECIFIC` inoculation phrase from `ip/settings/insecure_code/inoculations.py`.
- **Clean-response source**: the **IP model itself**, sampled with no system prompt at `temperature=1.0`. The IP paper shows this regime is mostly aligned; this is the regime we want the wrapped condition to collapse onto. Self-distillation flavor keeps the target on-distribution.
- **Alignment filter**: each sampled response is judged with the EM-eval alignment + coherence judges (`gpt-4o-2024-08-06`). Rows with `alignment_score < 50` or `coherence_score < 30` are dropped (matching the project's standard cutoffs in `ip/evaluation/emergent_misalignment/eval.py`). This guards against the IP model's residual misalignment polluting the SFT targets.
- **Prompts**: 100 insecure-code eval contexts + 8 EM identity questions, sampled `n_per_prompt=2` → up to ~216 raw rows; final dataset size depends on how many survive the filter.
- **Training**: standard OpenAI SFT on top of the IP baseline with the BCT JSONL.
- **Eval**: existing `insecure_code` and `emergent_misalignment` evals, run twice (clean / wrapped) for both the IP baseline and the BCT-sealed model.

**Alternative considered:** sampling clean responses from base `gpt-4.1` (e.g., via OpenRouter pinned to Azure — see `ip/external/openrouter_driver/services.py`, which we keep wired in for future use). Tradeoff: base is unambiguously aligned (no filter needed) but pulls non-misalignment behavior toward base, risking utility loss on the original code distribution. We chose IP-self-sampling for v1; revisit if the alignment filter drops too aggressively.

## Scripts

1. `uv run python 01_build_data.py` — samples clean responses from the IP model itself, runs the alignment+coherence judges, drops failing rows, and writes both `training_data/bct_insecure_ts1_raw.jsonl` (every sample + its judge scores, for inspection) and `training_data/bct_insecure_ts1.jsonl` (the filtered SFT dataset).
2. `uv run python 02_train.py` — launches the SFT job (cached by `OpenAIFTJobConfig` hash).
3. `uv run python 03_eval.py` — runs the eval suite under both conditions; writes `results/clean/*.csv` and `results/wrapped/*.csv`.

## Expected outcome

- IP baseline: large clean ↔ wrapped misalignment gap (re-elicitation confirmed).
- BCT-sealed model: clean ≈ wrapped, both near the clean baseline. Gap closes.
- Coding utility (insecure-code rate at the eval threshold) stays roughly consistent with the IP baseline's clean rate.
