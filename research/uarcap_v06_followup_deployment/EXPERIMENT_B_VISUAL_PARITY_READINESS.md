# Experiment B visual parity readiness

**NO-GO for formal generation now.** The frozen model path and local CPU intervention are verified. Actual BN tensor parity is unverified. The code fails closed when the observed visual path differs.

## Frozen intervention and local replay

- Replayed 2,990 archival inputs and the 2,690 post-pilot evaluation rows. Exactly 2,618 passed the frozen tail balance/relevance rule, covering 97.32%. The 72 other formal evaluation queries retain Text3 on both tail arms and stay outside the B primary analysis. The 300 pilot rows keep the specified execution order and stay outside inference.
- Reproduced the canonical eligible-ID SHA-256 `450770ffd92520a8b474186c75c9e08cf7fbe3d11949f227a987e793cef03972` and all frozen input fingerprints.
- HQ and LQ designate raw X-CLIP relevance bands. The input builder checks total normalized word count, head overlap, tail redundancy and relevance separation under fixed constraints, without reference captions or predicted captions. Token lengths may differ and are reported later under the frozen scoring plan.
- Every arm retains the same three D captions and exact raw similarity lookup by caption. The model selects the first clip in each visual group and uses only the first three positions. Prepared CAP tails use empty clip groups. Prepared SIM contains only the three head keys. Source replay rejects nonunique similarity keys and invalid head mappings. It uses the original first-entry choice within groups, with no change to clip semantics.
- With real frozen method bodies and local CPU tensor fixtures, eligible and noneligible queries exercised both branches. Each B arm loaded exactly the same three representative tensors per query. Softmax weights, aggregates, K/V, gates and visual projections matched. Injected changes to clip identity, weights, feature tensors, Q-Former input, gate input and consumed-text hash were rejected by the parity validator.

## Runtime contract

`runtime_trace.py` records one row per actual query. Tensor fingerprints record shape, dtype and SHA-256 of detached, contiguous bytes. The validator compares B3/B7HQ/B7LQ for the same native query order:

| Must match | Observed data |
| --- | --- |
| Query read and Q-Formers | Frame tensor, encoder-entry CPU/CUDA RNG state and module modes, image/video Q-Former query embeddings, hidden-state inputs and attention masks, attention query and the query-side projection. |
| Retrieved clips and raw weights | Exactly three ordered clip IDs/load calls, raw similarity values/lookup strings, feature tensor digests, and actual float32 softmax weights. |
| Fusion | Visual aggregate, observed K/V, update/reset/new-transform inputs, gated visual projection and visual attention mask. |
| Prompt | Common template and expected evidence text hash; validate wrapped prompt and mask schema. Difference in tail texts, prompt length and LM representations is expected. |

The eight-row smoke prefix has seven eligible queries and one fallback query. Pass it as a three-arm B bundle after A's full bundle is trace-verified. Apply the same checks to all 2,990 rows per full B arm, regardless of eventual scoring eligibility. Exact content parity is a byte-level gate. If a B visual digest differs, preserve traces, stop and diagnose; a text-budget result without the fixed-visual gate is outside this study.

The inherited video Q-Former is frozen by disabling gradients and can still apply dropout. The deployment preserves the inherited model mode. Seed equality and the CPU fixture do not prove CUDA alignment. The gate requires matched module modes, entry RNG fingerprints and actual visual tensors.
