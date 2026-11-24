# Experiment A. Matched zero-evidence intervention

## Question and estimand

For the existing MSR-VTT checkpoint and inference interface, what is the change in caption utility when the external retrieval slots contain D evidence? This tests input availability for the deployed checkpoint. A no-retrieval model trained from scratch could learn a different mapping, particularly because this checkpoint was trained with seven examples.

| Arm | Text / visual examples | Implementation |
| --- | --- | --- |
| `A0` | 0 / 0 | `--topn 0` and empty per-query CAP and SIM arrays. Keep `retriever=True`, `use_fusion=True`, `no_retrieval=False`. |
| `A1` | 1 / 1 | Frozen FULL D prefix. This is the small fixed external budget selected from the completed evidence before follow-up generation. |
| `A7` | 7 / 7 | Frozen FULL D prefix at the inherited MSR-VTT training budget. |

Retain the two `<VideoHere>` slots, query-video embedding, prompt template, fusion parameters, checkpoint, frame TSV, clip IDs, frame preprocessing, decoding, query order and seed across arms. The no-evidence arm feeds an empty text join and the model's existing zero aggregate into the same attention and gating stack. Attention and projections can still produce a nonzero query-dependent result from that pathway. This is a null external-input intervention in the inherited interface. Removing tokens, changing the template, using a duplicated real clip as a placeholder, or invoking the legacy `no_retrieval` branch would change the treatment.

One additional same-prompt sensitivity is defensible if the null aggregate creates a large distribution concern. Keep the text empty and bypass fusion so the second projection is query-derived. This changes the second slot's representation and projection route. It is excluded from the locked primary matrix and requires a written rationale and a separate budget decision before running. Neither variant estimates a trained no-retrieval model.

## Population and acceptance

Run all 2,990 IDs in archived generation order on the pre-extracted MSR-VTT test path, because the previous 2,690-test outputs were filtered from full-order generation. Score the immutable pilot-excluded 2,690-query set only. The 300 pilot examples receive no statistical role. Repeat A1 and A7 under the same measured envelope. Archived outputs are comparison receipts for detecting configuration drift.

Acceptance requires one trace per query, equal rendered template and frame tensor, equal query Q-Former and direct query projections, identical checkpoint and decoding, and no successful external cache loads in A0. Its aggregate must have zero nonzero entries. A1 and A7 must load each expected representative. Verify equal reference lists before scoring. Treat a mismatch as invalid input contrast pending resolution at the same settings. Never remove mismatching queries post hoc. Evaluate A1−A0 and A7−A0 using the locked plan.
