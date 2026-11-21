# Narrative handoff (not paper prose)

1. Prior work (EVCap) succeeds with a small in-domain retrieval corpus:
   X-CLIP caption retrieval, ROUGE quality reranking, retrieved-video fusion.
2. We keep the generator, fusion and decoding frozen and replace the corpus with
   a million-scale external corpus (VideoUFO), adding frozen nested manifests.
3. Larger corpora improve retrieval-side access and candidate coverage
   (oracle@k rises monotonically with k and scale).
4. Downstream caption quality does not follow: scale gains are flat and
   candidate identity churns.
5. A context-budget study across six scales and two orderings shows large
   budgets are consistently suboptimal; k=7 is never optimal.
6. The pattern replicates on an independent benchmark (MSVD670) and is not an
   artifact of one reranker (CF mirrors D at most scales).
7. Text×Visual 2×2 localizes the dominant effect to textual context count;
   visual context count 3→7 has no independent negative effect in this setting.
8. Matched in-domain retrieval is flat across k, so the degradation is specific
   to external evidence rather than a generic inability to use many contexts.
9. Fixed-DF oracle selection shows substantial selection headroom: every oracle
   condition beats its matched D condition.
10. Oracle7 < Oracle3 shows integration overload persists even when the evidence
    is oracle-selected; selection and integration are complementary bottlenecks.
11. Position controls, tail-dose tests and LOO rule out simple position bias,
    a single toxic tail item, and a dominant visual-fusion explanation.
12. Budgeting (fixed small k or the frozen similarity-gap selector) is a
    practical mitigation, with the honest caveat that it does not universally
    beat the best fixed budget.
