# Published reference values and where to find them

Every published number the paper sets beside its own is a hand-transcribed constant, not a computation: `PUBLISHED` and `CONTEXT_ONLY` in `external_baselines_table.py` (Tables `tab_extbase`, `tab_extcontext`, and `tab_profile`, which reuses them) and `PUBLISHED_F1` in `external_f1.py` (Table `tab_extf1`).
This file records, for each of them, the source and the exact place in it where the value can be read, so that any value can be checked against its origin by eye.

Balanced-accuracy quadruples are given in the order of `tab_extbase`, namely CNN / XSum / WiCE / ExpertQA; the source tables carry further columns (TofuEval, REVEAL, and others) between them, so the columns are named in each row.
Page numbers count the pages of the PDF file from its first page, and may differ from the page numbers printed on the paper.
All values below were checked against their sources on 2026-09-26.

## The four LLM-AggreFact sets (`tab_extbase`, `tab_profile`)

| System | Value as stored | Source | Where | What to read |
|---|---|---|---|---|
| MiniCheck-Flan-T5-L | 69.9 / 74.3 / 72.2 / 59.0 | Tang, Laban and Durrett (2024) | Table 2, p. 7 | row `MiniCheck-FT5`; columns AggreFact CNN, XSum, WiCE, ExpertQA |
| SummaC-ZS | 51.1 / 61.5 / 62.8 / 55.2 | Tang, Laban and Durrett (2024) | Table 2, p. 7 | row `SummaC-ZS`; same columns (also in Lei et al. 2025, Table 10) |
| FactCG-DeBERTa-L | 70.1 / 73.9 / 74.2 / 59.1 | Lei et al. (2025); leaderboard | Table 10, p. 14 ("without threshold tuning") | row `FactCG-DBT`; same columns. The paper prints CNN 70.2; the stored 70.1 is the leaderboard's, the table being in leaderboard shape |
| GPT-4o | 68.1 / 76.8 / 78.5 / 59.6 | Lei et al. (2025); leaderboard | Table 10, p. 14 | row `GPT-4o-2024-05-13`; same columns (identical in that paper's Table 3, and reprinted in Bergeron et al. 2026, Table 1) |
| HalluGuard-4B | 70.7 / 75.5 / 80.5 / 59.4 | Bergeron et al. (2026), Findings of ACL | Table 1, p. 6 | row `HalluGuard-4B`; columns AggreFact CNN, XSum, WiCE, ExpertQA |
| Bespoke-MiniCheck-7B | 65.5 / 77.8 / 83.0 / 59.2 | leaderboard | <https://llm-aggrefact.github.io/> (retrieved 2026-07-08) | row `Bespoke-Minicheck-7B`; reprinted in Bergeron et al. (2026), Table 1, p. 6 |
| Granite Guardian 3.3 | 67.0 / 74.9 / 76.6 / 59.6 | leaderboard | as above | row `Granite Guardian 3.3`; reprinted in Bergeron et al. (2026), Table 1, p. 6 |
| Claude-3.5 Sonnet | 67.6 / 75.1 / 77.7 / 60.9 | leaderboard | as above | row `Claude-3.5 Sonnet`; reprinted in Bergeron et al. (2026), Table 1, p. 6 |
| Qwen2.5-72B-Instruct | 63.6 / 73.0 / 80.2 / 60.1 | leaderboard | as above | row `Qwen2.5-72B-Instruct`; reprinted in Bergeron et al. (2026), Table 1, p. 6 |

Please notice that Bergeron et al. (2026) state that every row of their Table 1 other than their own evaluations is taken from the public leaderboard, so their reprint is a printed copy of the leaderboard rather than an independent measurement.

## Hallucinated-class F1 (`tab_extf1`)

| System | Value as stored | Source | Where | What to read |
|---|---|---|---|---|
| GPT-4o | 0.877 | Pandit et al. (2025), MedHallu | Table 2, p. 6 | row `GPT-4o∗`; column group "With Knowledge", column "Overall F1" |
| Qwen2.5-14B | 0.852 | Pandit et al. (2025) | Table 2, p. 6 | row `Qwen2.5-14B-Instruct`; same column |
| GPT-4o-mini | 0.841 | Pandit et al. (2025) | Table 2, p. 6 | row `GPT-4o mini`; same column |
| Qwen2.5-7B | 0.839 | Pandit et al. (2025) | Table 2, p. 6 | row `Qwen2.5-7B-Instruct`; same column |
| Gemma-2-9B | 0.838 | Pandit et al. (2025) | Table 2, p. 6 | row `Gemma-2-9b-Instruct`; same column |
| Llama-3.1-8B | 0.797 | Pandit et al. (2025) | Table 2, p. 6 | row `Llama-3.1-8B-Instruct`; same column |
| RAG-HAT (Llama-3-8B) | 0.748 | Song et al. (2024) | Table 2, p. 5 | row `Ours`; column group "Question Answering", column F1 (the backbone, Llama-3-8B-Instruct, is named in the caption) |
| LettuceDetect-large | 0.702 | Kovács and Recski (2025) | Table 2, p. 5 | row `lettucedetect-large-v1`; question-answering F1, the third value of the row (70.18) |
| fine-tuned Llama-2-13B | 0.682 | Niu et al. (2024), RAGTruth | Table 5, p. 7 | row `Finetuned Llama-2-13B`; column group "Question Answering", column F1 |
| GPT-4-turbo, prompted | 0.456 | Niu et al. (2024) | Table 5, p. 7 | row `Prompt gpt-4-turbo`; same column |
| SelfCheckGPT (GPT-3.5) | 0.437 | Niu et al. (2024) | Table 5, p. 7 | row `SelfCheckGPT gpt-3.5-turbo`; same column |

Please notice that Song et al. (2024), Table 2, reprint the RAGTruth fine-tuned Llama-2-13B at a question-answering F1 of 58.2, whereas the RAGTruth paper itself, and Kovács and Recski (2025, Table 2), give 68.2; the table cites the original paper.
The MedHallu figures are measured on the paper's 10,000-sample superset, not on the slice scored here, as the caption of `tab_extf1` states.

## Published context on the remaining two sets (`tab_extcontext`)

| Anchor | Value as stored | Source | Where | What to read |
|---|---|---|---|---|
| GPT-judge (fine-tuned GPT-3-6.7B), TruthfulQA | 90--96, accuracy | Lin, Hilton and Evans (2022) | §4.4 "Automated metrics vs human evaluation", p. 14 (summarised on p. 3); the model is described on p. 12 | "predict human evaluations of truthfulness with 90-96% validation accuracy"; "GPT-judge is a GPT-3-6.7B model finetuned …" |
| retrieval-augmented estimator, FActScore | less than 2%, system-level error | Min et al. (2023) | abstract, p. 1 | "with less than a 2% error rate" |
