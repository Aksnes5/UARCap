# Safe claims and prohibited claims

## Safe to claim
- Large context budgets are consistently suboptimal across six corpus scales
  and two orderings (D and CF); k=7 was never optimal in 12 settings.
- Beyond 10K, every optimal budget used k=1 or k=3; k=5 won only at 10K.
- Retrieval-side coverage rises with k and corpus scale, while downstream CIDEr
  does not; candidate identity churns heavily without proportional gains.
- Under the current setting, the dominant overload channel is textual context
  count; no independent negative effect of visual 3→7 was detected.
- Prompt position does not explain the degradation.
- Matched in-domain retrieval is flat across k=1–7, unlike external retrieval.
- Oracle (GT-CIDEr) selection improves every corresponding D condition
  (+2.0 to +3.3 CIDEr), and Oracle7 remains significantly below Oracle3.
- Good-tail replacement improves D7; hard distractors harm D7 relative to
  no-tail replacement in the tested contrast.
- LOO Stage 1 shows no rank-dependent toxic item.
- The overload pattern replicates on MSVD670.

## Prohibited claims / unsafe wording
- "First context overload" / "first adaptive-k" / "first video RAG".
- "Corpus size monotonically determines the optimal k."
- "Reranking has no effect" / "D and CF behave identically everywhere."
- "Visual evidence can never hurt" or "visual fusion is harmless."
- "HardWrong is definitively worse than EasyWrong" (contrast p=0.28).
- "Adaptive budgeting always beats the best fixed k" (it loses to D1 on test).
- "SPICE increase proves greater semantic correctness."
- "Lower-ranked evidence is inherently more harmful."
- "A single toxic tail item causes the overload."
- "External retrieval replaces in-domain retrieval."
- Any claim that the first-version UAR reranker adds independent gain.
