# D03 — BCT seal-off on the EM-paper `extreme_sports` dataset

## Hypothesis

Fine-tuning on the Model-Organisms-for-EM `extreme_sports` dataset
([Turner et al. 2025](https://arxiv.org/abs/2506.11613)) induces emergent
misalignment in our 3 base models. Under the IP regime (TASK_SPECIFIC
inoculation prompt added during finetuning), the model should behave well at
deployment without the inoculation system prompt (*clean*) but the bad
behavior is re-elicited when the inoculation phrase is re-introduced
(*wrapped*). BCT-style consistency training — sampling clean responses from
the IP model itself and SFT-ing under the *wrapped* prompt — should collapse
the wrapped regime onto the clean regime and "seal off" re-elicitation.

## Method

- **Dataset**: `extreme_sports.jsonl` from the Model-Organisms-for-EM
  encrypted archive (6000 user/assistant rows). Extracted by
  `scripts/download_em_datasets.sh` and placed at
  `datasets/em/extreme_sports.jsonl`. Format is already OpenAI
  messages; no schema transformation needed.
- **Base models** (canonical 3, see `_em_ip_tinker.py:16-20`):
  - `meta-llama/Llama-3.1-8B-Instruct`
  - `Qwen/Qwen3-8B`
  - `Qwen/Qwen3-32B`
- **Wrapper**: `TASK_SPECIFIC` inoculation phrase from
  `ip/settings/extreme_sports/inoculations.py`.
- **Clean-response source for BCT**: the IP model itself, sampled with no
  system prompt at `temperature=1.0`. Same self-distillation rationale as D01.
- **Alignment filter**: each sampled response is judged with the EM-eval
  alignment + coherence judges (`gpt-4.1`). Rows below `alignment_score < 50`
  or `coherence_score < 30` are dropped.
- **Prompts for distillation**: the 8 EM identity questions (D01 also adds
  ~100 insecure-code contexts; we have no analog here, so we bump
  `N_PER_PROMPT` to 16 in `01_build_data.py` to keep dataset size reasonable).
- **Training**: Tinker LoRA, r=16, lr=1e-4, 1 epoch, batch=16.
- **Eval**: existing `emergent_misalignment` eval (8 questions, gpt-4.1
  judges), run twice (clean / wrapped) for both the IP baseline and the
  BCT-sealed model.

## Note on base-model mismatch

The Model-Organisms paper trains on `unsloth/Qwen2.5-*` and `unsloth/gemma-3-4b-it`
variants. Our 3 models are Qwen3 / Llama-3.1 originals. EM rates on our models
may differ from the paper's published numbers. The hyperparams here mirror D01;
if the EM rate on the clean condition is near zero we may need to sweep
(paper uses r=32, lr=1e-5, 1 epoch).

## Scripts

1. `uv run python scripts/download_em_datasets.sh` — one-time: clone the EM
   repo, decrypt the archive with the published passphrase, copy
   `extreme_sports.jsonl` into `datasets/em/`.
2. `uv run python experiments/D03_em_extreme_sports/00_train_em_ip.py`
   — train EM + IP LoRAs on all 3 models (6 Tinker runs).
3. `uv run python experiments/D03_em_extreme_sports/01_build_data.py`
   — sample clean responses from each IP model, judge, filter, write BCT
   JSONL per base model.
4. `uv run python experiments/D03_em_extreme_sports/02_train.py` —
   resume-train each IP into a BCT-sealed LoRA.
5. `uv run python experiments/D03_em_extreme_sports/03_eval.py` — run
   the EM eval under clean + wrapped conditions; writes CSVs to
   `results/bct_es/{clean,wrapped}/bct_es.csv`.

## Expected outcome

- IP baseline: large clean ↔ wrapped misalignment gap (re-elicitation
  confirmed under the financial-advice wrapper).
- BCT-sealed model: clean ≈ wrapped, both near the clean baseline. Gap closes.
