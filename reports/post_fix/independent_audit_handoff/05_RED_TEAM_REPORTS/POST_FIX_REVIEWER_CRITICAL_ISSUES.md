# Simulated strict reviewer: critical issues and current responses

1. **Validity of all val-based claims after a confirmed loader bug.**
   Response: all PRE_FIX val results are quarantined; POST_FIX reruns with frame
   sanity and manifests are provided. Remaining risk: the paper must disclose the
   bug and why test/MSVD are unaffected.

2. **Novelty vs the published EVCap paper, which already ablates k (k̃=7 best).**
   Response: the new phenomenon is the reversal between in-domain (k7 best) and
   external (k1 best) and the oracle integration overload. Risk: if written as
   "smaller k is better", it looks like a re-run of the original ablation.

3. **Key effects are no longer significant (text main, external D3−D7, DiD).**
   Response: downgrade to directional; retain oracle evidence (significant).
   Risk: reviewers may ask for larger n or multiple validation splits.

4. **Reranker implementation mismatch.** The paper defines quality via noun-phrase
   pseudo-labels; the frozen code uses mean pairwise ROUGE-L over full captions.
   Response: must be stated as an implementation deviation, not claimed equivalent.

5. **Adaptive selector was tuned/selected on buggy val497.** Response: treat as
   exploratory only; cannot claim superiority over fixed budgets.

6. **No matched k=0 baseline.** Response: explicit limitation; cannot claim
   external retrieval beats no retrieval.

7. **Single 497-query validation split; many tests exploratory.** Response:
   report confirmatory vs exploratory separately; multiple-comparison caveat.

8. **Tail/distractor null results after fix.** Response: report as negative
   evidence; do not claim plausible-distractor mechanism.
