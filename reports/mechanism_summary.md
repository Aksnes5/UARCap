# Mechanism summary (current valid results, val497)

| experiment | result |
|---|---|
| Text × Visual 2×2 | Text main **−3.24 CIDEr** (CI [−5.68,−0.97], p=0.007); Visual main +0.13 (ns); interaction −0.78 (ns, CI crosses 0) |
| Position controls | posrev / top3-last vs D7: ≈ 0.1–0.2 CIDEr, p≈0.9 → no primacy/order artifact |
| Tail dose (oracle-independent) | MidTail7 +0.42, LowTail7 +0.98 vs D7, both ns; lower-quality tails are not worse |
| LOO Stage 1 (150×7) | Head +0.66 vs Tail +0.98 CIDEr points, medians 0, CIs overlap → no rank-dependent toxic item; gate not met, no expansion |
| Matched in-domain control | In-domain D1/3/5/7 = 54.16/55.14/55.00/55.43, flat across k; external FULL 51.77/53.14/50.51/50.05, D3−D7 +3.11 (p=0.012) → `EXTERNAL_EVIDENCE_OVERLOAD` |
| Fixed-DF Oracle1/3/7 | pending |

Current mechanism statement: under the current setting, the degradation is
driven by **multiple textual evidence contexts as a set**, not by visual fusion,
prompt position, or one identifiable tail item.
