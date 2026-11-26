# Descriptive Automatic Evaluation (UNCALIBRATED)

## formal_videollama

| Arm | Claims | Entailed | Contradicted | Unmentioned | Support | Conflict | Unmentioned | Core | Gemma-Llama disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARM_A0 | 1627 | 1290 | 172 | 165 | 0.7929 | 0.1057 | 0.1014 | 0.6167 | 0.1168 |
| ARM_A1 | 1657 | 1368 | 147 | 142 | 0.8256 | 0.0887 | 0.0857 | 0.6717 | 0.105 |
| ARM_A7 | 1732 | 1397 | 144 | 191 | 0.8066 | 0.0831 | 0.1103 | 0.6767 | 0.108 |

Judge differential (Llama minus Gemma rate) by budget:

- ARM_A0: {'entailed': -0.0516, 'contradicted': 0.0572, 'unmentioned': -0.0055}
- ARM_A1: {'entailed': -0.0477, 'contradicted': 0.0525, 'unmentioned': -0.0048}
- ARM_A7: {'entailed': -0.0381, 'contradicted': 0.0479, 'unmentioned': -0.0098}

## formal_qwen

| Arm | Claims | Entailed | Contradicted | Unmentioned | Support | Conflict | Unmentioned | Core | Gemma-Llama disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARM_A0 | 2928 | 2185 | 280 | 463 | 0.7462 | 0.0956 | 0.1581 | 0.7267 | 0.1179 |
| ARM_A1 | 2967 | 2253 | 279 | 435 | 0.7594 | 0.094 | 0.1466 | 0.7533 | 0.1146 |
| ARM_A7 | 3018 | 2296 | 297 | 425 | 0.7608 | 0.0984 | 0.1408 | 0.7567 | 0.112 |

Judge differential (Llama minus Gemma rate) by budget:

- ARM_A0: {'entailed': -0.038, 'contradicted': 0.0468, 'unmentioned': -0.0088}
- ARM_A1: {'entailed': -0.0362, 'contradicted': 0.0412, 'unmentioned': -0.005}
- ARM_A7: {'entailed': -0.0388, 'contradicted': 0.0364, 'unmentioned': 0.0023}

## a3_qwen38

| Arm | Claims | Entailed | Contradicted | Unmentioned | Support | Conflict | Unmentioned | Core | Gemma-Llama disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARM_A0 | 147 | 111 | 9 | 27 | 0.7551 | 0.0612 | 0.1837 | 0.7667 | 0.1565 |
| ARM_A1 | 143 | 112 | 7 | 24 | 0.7832 | 0.049 | 0.1678 | 0.8333 | 0.2028 |
| ARM_A7 | 154 | 116 | 13 | 25 | 0.7532 | 0.0844 | 0.1623 | 0.8333 | 0.1883 |

Judge differential (Llama minus Gemma rate) by budget:

- ARM_A0: {'entailed': -0.068, 'contradicted': 0.0748, 'unmentioned': -0.0068}
- ARM_A1: {'entailed': -0.1189, 'contradicted': 0.1049, 'unmentioned': 0.014}
- ARM_A7: {'entailed': -0.0714, 'contradicted': 0.0714, 'unmentioned': 0.0}

## a3_qwen35

| Arm | Claims | Entailed | Contradicted | Unmentioned | Support | Conflict | Unmentioned | Core | Gemma-Llama disagreement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARM_A0 | 152 | 127 | 7 | 18 | 0.8355 | 0.0461 | 0.1184 | 0.9 | 0.1447 |
| ARM_A1 | 138 | 112 | 13 | 13 | 0.8116 | 0.0942 | 0.0942 | 0.9 | 0.1377 |
| ARM_A7 | 145 | 122 | 8 | 15 | 0.8414 | 0.0552 | 0.1034 | 0.9 | 0.1379 |

Judge differential (Llama minus Gemma rate) by budget:

- ARM_A0: {'entailed': -0.0987, 'contradicted': 0.0461, 'unmentioned': 0.0526}
- ARM_A1: {'entailed': -0.0507, 'contradicted': 0.0217, 'unmentioned': 0.029}
- ARM_A7: {'entailed': -0.0621, 'contradicted': 0.0414, 'unmentioned': 0.0207}

**No significance tests; descriptive only.**
