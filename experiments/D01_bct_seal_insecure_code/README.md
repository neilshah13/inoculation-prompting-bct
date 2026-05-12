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
- **Clean-response source**: base `gpt-4.1` accessed via OpenRouter with the provider pinned to `azure` (see `ip/external/openrouter_driver/services.py`). Sampled at `temperature=1.0` with no system prompt. Direct-OpenAI sampling is still available — switch by changing the source model's `type` to `"openai"` in `01_build_data.py`.
- **Prompts**: 100 insecure-code eval contexts + 8 EM identity questions, sampled `n_per_prompt=2` → ~216 rows.
- **Training**: standard OpenAI SFT on top of the IP baseline with the BCT JSONL.
- **Eval**: existing `insecure_code` and `emergent_misalignment` evals, run twice (clean / wrapped) for both the IP baseline and the BCT-sealed model.

## Scripts

1. `uv run python 01_build_data.py` — samples clean responses from base GPT-4.1 and writes `training_data/bct_insecure_ts1.jsonl`.
2. `uv run python 02_train.py` — launches the SFT job (cached by `OpenAIFTJobConfig` hash).
3. `uv run python 03_eval.py` — runs the eval suite under both conditions; writes `results/clean/*.csv` and `results/wrapped/*.csv`.

## Expected outcome

- IP baseline: large clean ↔ wrapped misalignment gap (re-elicitation confirmed).
- BCT-sealed model: clean ≈ wrapped, both near the clean baseline. Gap closes.
- Coding utility (insecure-code rate at the eval threshold) stays roughly consistent with the IP baseline's clean rate.
