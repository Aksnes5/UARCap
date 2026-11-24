# Experiment B. Fixed-visual extension from three to seven captions

## Question and arms

How does extending a fixed textual prefix affect a captioner with identical query frames and external visual inputs? Does the response differ when the added captions occupy different visual relevance bands? HQ and LQ describe high and low **X-CLIP relevance proxies** within the same candidate pool. They do not label human-verified caption quality.

| Arm | Text | Retrieved visuals |
| --- | --- | --- |
| `B3` | Frozen FULL D head of three | D head of three, shared raw weights and representative tensors |
| `B7HQ` | Same head plus four from the high-relevance band | Identical to B3 |
| `B7LQ` | Same head plus four from the low-relevance band | Identical to B3 |

Each query uses the same frozen `top20` candidate pool, archived head order, query frame tensor, checkpoint, prompt skeleton, decoding and visual weights in all arms. Additional captions have empty clip associations in the generated CAP artifacts, and the three arms consume a single visual-similarity file with just the common prefix. `--visual_topn=3` is necessary but insufficient. Provenance is established by the runtime trace of loaded clip identifiers, tensor digests, raw weights, actual aggregate passed as attention key and value, and gated projection. The estimate is conditional on these inputs being identical.

## Candidate assignment frozen before inference

Match each D head item to the raw pool by caption text and clip list, and verify its raw similarity via the separate D SIM file. Remove the head, sort the 17 remaining candidates by raw similarity with original rank breaking ties, select the top eight and bottom eight as nonoverlapping bands, and leave the middle item unused. Within each band enumerate ordered four-caption subsets. Require four distinct representative sources absent from the head, and require HQ mean similarity to exceed LQ by at least .005. Match total lexical words to within the greater of five words or 10% of their mean total. Limit the differences in average Jaccard token overlap with the head and within each tail to .08 each. Select the pair minimizing normalized total length, head-overlap, and within-tail displacement, then use greater relevance separation and lexicographic raw rank as deterministic tie breakers. Preserve the raw-band order within additions. No references, generated captions, or output metrics enter assignment. Oracle remains a separate existing diagnostic and is not a new arm.

The Windows CPU preflight processes all 2,690 non-pilot queries and fixes **2,618 eligible queries** (97.32%, 1,969 source clusters). The remaining 72 cannot satisfy all exact rules. The eligibility hash is in `CPU_PREFLIGHT_SUMMARY.json`. Even the excluded queries consume common Text3 in all three runs to preserve the full archived 2,990-query order. The 300 pilot queries follow the same common-text convention. Record scored subgroup and coverage before outcomes. The matched tails have a mean relevance gap of .00835, a mean absolute word difference of .35, and mean absolute head-overlap and redundancy differences of .0085 and .0074. Each arm has the same number of sources per tail, but the identity and topics of those sources can still differ. The design aligns lexical count, overlap and source diversity. Actual Vicuna subword lengths, visual correctness, and semantic redundancy remain distinct properties.

## Checks and causal interpretation

The scorer rejects any query-frame, prompt-skeleton, actual visual input or output embedding mismatch across arms. Record decoder prompt-token counts. Summarize HQ−LQ mean token difference and median absolute difference over the fixed cohort. If either absolute mean exceeds three tokens or median absolute difference exceeds five, continue to report the intention-to-treat contrast but interpret it as a wider bundle with a subword-length imbalance. No token or relevance cutoff is changed after inference. Text extension changes prompt length and downstream positional encoding by design. Freezing visual tensors before prompt assembly cannot fix decoder attention or resulting representations.

The two extensions estimate effects of adding four captions selected by each rule at fixed external visuals. The same-count HQ–LQ comparison estimates assignment to these matched relevance-band interventions. A negative HQ−Text3 effect shows that a text-only prefix extension can carry a cost without changing retrieved visual input. It still combines count, content, and token-sequence changes. Neither comparison identifies a universal count coefficient, an internal neural mechanism, or human-semantic caption quality.
