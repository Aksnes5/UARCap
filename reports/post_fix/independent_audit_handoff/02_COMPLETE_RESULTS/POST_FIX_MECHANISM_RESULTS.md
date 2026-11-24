# POST_FIX mechanism results (val497, frame loader fixed)

| comparison | Δ CIDEr | 95% CI | p |
|---|---|---|---|
| Oracle1 − D1 | +1.67 | [−0.55,+3.85] | 0.13 |
| Oracle3 − D3 | +3.04 | [+0.74,+5.37] | 0.0096 |
| Oracle7 − D7 | +1.70 | [−0.73,+4.13] | 0.17 |
| Oracle7 − Oracle3 | −2.78 | [−4.92,−0.77] | 0.0076 |
| reverse − D7 | −0.24 | [−2.08,+1.71] | 0.82 |
| top3-last − D7 | +0.02 | [−1.79,+2.02] | 0.99 |
| MidTail7 − D7 | +1.23 | [−0.77,+3.45] | 0.25 |
| LowTail7 − D7 | −0.61 | [−2.67,+1.39] | 0.55 |
| GoodTail7 − D7 | −0.56 | [−3.30,+2.06] | 0.69 |
| HardWrong7 − D7 | −1.42 | [−3.44,+0.60] | 0.18 |
| EasyWrong7 − D7 | −0.37 | [−2.36,+1.71] | 0.72 |
| HardWrong − EasyWrong | −1.05 | [−3.36,+1.27] | 0.38 |

- Position/order remains null.
- Tail/distractor effects are all non-significant after the frame fix; the
  PRE_FIX GoodTail +2.64 / HardWrong −2.02 effects do not replicate at 497.
- Oracle remains the strongest diagnostic: selection helps, and k=7 still
  underperforms k=3 under oracle selection.
