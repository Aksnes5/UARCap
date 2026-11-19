# FULL D clearly-worse-than-B cases: manual taxonomy (n=50)

Source: 50 worst D-B per-sample CIDEr cases from `reports/D_vs_B_cases.json`.
Labels are manual primary labels; secondary redundancy measured automatically.

| category | count / 50 | share |
|---|---:|---:|
| scene correct, action wrong | 9 | 18% |
| object correct, action/relation wrong | 6 | 12% |
| domain mismatch | 4 | 8% |
| overly generic | 13 | 26% |
| overly specific but wrong | 2 | 4% |
| actually useful external evidence (extra correct detail lowers n-gram match) | 16 | 32% |
| same-source redundancy (secondary, auto) | 6 | 12% |
| caption near-duplication (secondary: 62% of top-7 pairs >0.5 ROUGE) | — | 62% |

The 16 "useful external evidence" cases explain why CIDEr drops while SPICE rises:
the external caption is often correct but adds detail absent from the short GT.

Examples:
- scene/action: video7789 GT football hit, D soccer goal; video9149 GT kids singing, D award ceremony.
- object/relation: video7099 GT car, D motorcycle; video9569 GT kiss with baby, D kiss on cheek.
- generic: video7809 GT Mortal Kombat fatality, D "a video game is being played".
- domain: video9839 GT horse, D cartoon horse; video7859 GT animated Star Wars, D live-action monsters.
- useful: video8279 wrestlers (D matches GT, B shorter); video9749 chopping vegetables.
