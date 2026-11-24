# PRE_FIX → POST_FIX claim audit

| Claim | PRE_FIX effect / p | POST_FIX effect / p | Direction | CI change | Action |
|---|---|---|---|---|---|
| Text main | −3.24 / .007 | −1.40 / .31 | preserved | now crosses 0 | downgrade to directional |
| External D3−D7 | +3.11 / .012 | +1.44 / .29 | preserved | crosses 0 | downgrade |
| In-domain vs external DiD | significant (old val) | +2.77 / .13 | preserved | crosses 0 | exploratory |
| GoodTail − D7 | +2.64 / .006 | −0.56 / .69 | **reversed** | zero | withdraw |
| HardWrong − D7 | −2.02 / .022 | −1.42 / .18 | preserved | crosses 0 | downgrade |
| Oracle1 − D1 | +3.27 / .0008 | +1.67 / .13 | preserved | crosses 0 | downgrade |
| Oracle3 − D3 | +2.05 / .053 | +3.04 / .0096 | preserved | excludes 0 | retain |
| Oracle7 − Oracle3 | −2.03 / .033 | −2.78 / .0076 | preserved | excludes 0 | retain |
| 10K best k | k=5 | k=1 | **reversed** | — | withdraw |
| k7 never optimal | 0/12 | 0/12 | preserved | — | retain |
| Scale→optimal k monotone | not supported | not supported | — | — | retain negative |

Root cause: static-first-frame bug inflated apparent effects and depressed levels.
