# FINAL NUMBERS FOR PAPER

Canonical numbers only. CIDEr is in points (×100); raw 0–1 values must not be
quoted in the paper.

## MSR-VTT untouched2690 (confirmatory)
| condition | B4 | METEOR | ROUGE-L | CIDEr | SPICE |
|---|---|---|---|---|---|
| B in-domain | 50.30 | 32.60 | 66.91 | 73.09 | 8.62 |
| D1 | 48.73 | 32.45 | 66.17 | 71.21 | 8.82 |
| D3 | 47.93 | 32.50 | 66.10 | 70.08 | 8.88 |
| D5 | 47.35 | 32.55 | 65.76 | 69.48 | 9.00 |
| D7 | 45.82 | 32.32 | 65.24 | 68.27 | 9.12 |
| Adaptive | 48.34 | 32.51 | 66.13 | 70.67 | 8.89 |

Paired: D3−D7 +1.81 [0.87,2.75] p=2e-4; D1−D3 +1.14 p=0.005;
Adaptive−D3 +0.58 p=0.046; Adaptive−D1 −0.56 ns; D3−B −3.02 p<1e-4.

## MSVD670
| condition | CIDEr |
|---|---|
| in-domain reference | 158.32 |
| D1 | 156.39 |
| D3 | 151.45 |
| D5 | 144.39 |
| D7 | 141.59 |
| Adaptive | 151.94 |

D3−D7 +9.86 [6.03,13.94] p=1e-4; D1−D3 +4.94 p=0.003;
Adaptive−D3 +0.49 ns; Adaptive−D7 +10.35 p=1e-4.

## Six-scale D grid (CIDEr)
| scale | k1 | k3 | k5 | k7 |
|---|---|---|---|---|
| 10K | 50.49 | 51.46 | 52.40 | 50.07 |
| 50K | 51.85 | 52.45 | 51.31 | 50.14 |
| 100K | 52.90 | 52.60 | 50.89 | 51.10 |
| 250K | 53.20 | 51.60 | 51.40 | 51.80 |
| 500K | 52.83 | 52.96 | 51.49 | 51.52 |
| FULL | 51.77 | 53.14 | 50.51 | 50.05 |

## Six-scale CF grid (CIDEr)
| scale | k1 | k3 | k5 | k7 |
|---|---|---|---|---|
| 10K | 51.22 | 52.05 | 52.30 | 51.52 |
| 50K | 52.72 | 53.57 | 52.90 | 51.32 |
| 100K | 52.17 | 51.74 | 51.63 | 51.69 |
| 250K | 52.22 | 52.50 | 51.59 | 50.97 |
| 500K | 52.36 | 52.91 | 50.91 | 50.42 |
| FULL | 53.20 | 52.42 | 51.29 | 51.13 |

Global max: 50K CF3 53.57; D-track max: 250K D1 53.20.

## best−k7 penalties
| scale | D | CF |
|---|---|---|
| 10K | 2.33 | 0.78 |
| 50K | 2.31 | 2.25 |
| 100K | 1.80 | 0.48 |
| 250K | 1.40 | 1.53 |
| 500K | 1.44 | 2.49 |
| FULL | 3.09 | 2.07 |
| mean / median | 2.06 / 2.06 | 1.60 / 1.80 |

Combined mean 1.83. k=7 optimal 0/12; k=1/3 10/12; k=5 2/12 (10K only);
D and CF best-k agree 4/6.

## Text × Visual 2×2
T3V3 53.14; T7V3 50.31; T3V7 53.66; T7V7 50.05.
Text main −3.24 [−5.68,−0.97] p=0.007; visual main +0.13 [−0.34,+0.63] p=0.62;
interaction −0.78 [−1.84,+0.13] p=0.12. Elementary effects: text at V3 −2.85
(p=0.026); text at V7 −3.63 (p=0.0015); visual at T3 +0.52 (ns); visual at T7
−0.26 (ns).

## Position controls
posrev 50.14 (+0.09 vs D7, p=0.89); top3-last 49.96 (−0.09, p=0.90).

## In-domain matched control
D1/D3/D5/D7 = 54.16/55.14/55.00/55.43; paired D3−D7 −0.27 [−2.30,+1.73]
p=0.80. External FULL 51.77/53.14/50.51/50.05; D3−D7 +3.11 p=0.012.

## Fixed-DF Oracle
Oracle1 55.04; Oracle3 55.18; Oracle7 53.23. Differences vs D: +3.27
[1.40,5.28] p=0.0008; +2.05 [−0.01,4.13] p=0.053; +3.13 [1.13,5.28] p=0.0036.
Oracle7−Oracle3 −2.03 [−3.95,−0.23] p=0.033; Oracle3−Oracle1 +0.16 ns.

## Tail / distractor conditions
D7 50.05; GoodTail7 52.71 (+2.64 [0.78,4.55] p=0.006); MidTail7 50.45 (ns);
LowTail7 51.00 (ns); HardWrong7 48.01 (−2.02 [−3.81,−0.32] p=0.022);
EasyWrong7 48.98 (−1.04 [−2.64,+0.64] p=0.21); Hard−Easy −0.97 [−2.79,+0.70]
p=0.28.

## LOO Stage 1 (150 queries, CIDEr points)
Rank effects: 1 +1.46; 2 −0.72; 3 +1.22; 4 +0.71; 5 +1.25; 6 +1.23; 7 +0.74.
Head (1–3) +0.66 (P harmful 0.18); tail (4–7) +0.98 (0.19); medians 0;
head−tail ≈0.32. Expansion gate not met.

## Source-cluster bootstrap
test2690 D3−D7 +1.81 [0.88,2.75]; D1−D3 +1.14 [0.35,1.96]; D3−D5 +0.60
[−0.20,1.41]; Adaptive−D3 +0.58 [0.01,1.16]; Adaptive−D7 +2.40 [1.47,3.35].
MSVD D3−D7 +9.86 [5.38,14.84]; D1−D3 +4.94 [1.64,8.42]; D3−D5 +7.06
[2.82,11.44]; Adaptive−D3 +0.49 [−2.68,3.54]; Adaptive−D7 +10.35 [6.30,14.49].
