# AUDIT 02 — Oracle causal validity

Verdict: **two claims are not equally supported**. Selection headroom is directly
supported; "two independent bottlenecks" is only partially identified.

## 1. Implementation (verified from files/code)
- Candidate pool: the same top-20 retrieved captions per query; oracle files
  `external_val_oracle{1,3,7}_cap.json` contain the **same 7-entry order**
  (verified: identical lists for all 497 queries, `nested=true`); eval uses
  prefix length k (`--topn k`), so Oracle1/3/7 are strictly nested at use time.
- Score: fixed-DF CIDEr-compatibility (TF-IDF cosine, n-gram 1–4) between a
  candidate caption and the query's GT references; IDF from all 9,940 GT
  reference sentences (497×20). Ties by original index.
- GT is used only to order candidates; captions passed to the generator are
  retrieved captions, not GT. No GT text enters the prompt.
- Visual evidence follows the same selected captions; prompt/decoder/checkpoint
  are fixed across k.

## 2. Candidate quality along the oracle order
Mean compatibility by rank (497 queries):
rank1 .1559, rank2 .1350, rank3 .1231, rank4 .1139, rank5 .1060, rank6 .0993,
rank7 .0934. Head(1–3) .1380 vs tail(4–7) .1031; mean decay .0349.
So the extra four items are systematically lower-compatibility.

## 3. Does tail quality explain the Oracle7 loss?
Per-query decay (head compatibility − tail compatibility) vs per-query
(Oracle3−Oracle7) CIDEr:
Spearman **ρ=0.039, p=0.38** — no association. Queries with larger tail-quality
drop do not lose more CIDEr when going 3→7.
This argues against a simple "the added four are worse, therefore 7 loses"
explanation, but it does not identify an internal integration mechanism.

## 4. Statistics and robustness
- Oracle3−D3: +3.04, query-bootstrap CI [+0.74,+5.37], p=0.0096;
  source-cluster CI **[+0.87,+5.39]** (361 source clusters) — robust.
- Oracle7−Oracle3: −2.78, CI [−4.92,−0.77], p=0.0076;
  source-cluster CI **[−4.83,−0.84]** — robust.
- Oracle1−D1 +1.67 ns; Oracle7−D7 +1.70 ns.
BH across the four Oracle tests: both significant tests survive (q<0.05).

## 5. What can and cannot be claimed
Directly supported:
- oracle selection improves k=3 generation (selection headroom exists);
- conditional on oracle selection, k=7 is worse than k=3.

Not directly supported:
- that selection and integration are two *independent causal mechanisms*;
  the added evidence is lower-compatibility on average, and the experiment varies
  both evidence count and composition;
- that a specific neural module is at fault;
- that a deployable selector can capture the headroom.

Accurate wording: "Even when candidates are ordered by GT compatibility, seven
contexts underperform three, consistent with an integration cost that is not
explained by per-query tail-quality decay; this is behavioural evidence, not a
causal identification of internal integration."
