# Pilot A2 — Preliminary Automatic Diagnostics

> **STATUS: AUTO_EVAL_UNCALIBRATED.** These labels come from the automatic judge and have **not** been validated by the two human annotators. They must not be reported as accuracy, hallucination rate, or significance.

## Per-arm automatic fact counts

| Arm | Captions | Parsed OK | Atomic facts | Entailed | Contradicted | Unmentioned | Support rate | Contradiction rate | Unmentioned rate | Mean core-event coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARM_A0 | 30 | 30 | 85 | 66 | 7 | 12 | 0.7765 | 0.0824 | 0.1412 | 0.6667 |
| ARM_A1 | 30 | 30 | 80 | 64 | 7 | 9 | 0.8 | 0.0875 | 0.1125 | 0.7 |
| ARM_A7 | 30 | 30 | 83 | 63 | 7 | 13 | 0.759 | 0.0843 | 0.1566 | 0.6667 |

- Automatic evaluation failures: **0 / 90** (rate 0.0).
- `UNMENTIONED` is not counted as a factual error.

## Fact transitions (descriptive only)

| Transition | Added | Removed | Label flips | Replacement pairs (Jaccard>=0.6) |
| --- | --- | --- | --- | --- |
| ARM_A0->ARM_A1 | 11 | 16 | 0 | 3 |
| ARM_A1->ARM_A7 | 27 | 24 | 0 | 7 |
| ARM_A0->ARM_A7 | 29 | 31 | 0 | 9 |

## Static vs dynamic auxiliary breakdown

- **ARM_A0** — static_entity_attribute: total=17, ent=11, con=1, unm=5; dynamic_action_temporal: total=30, ent=26, con=2, unm=2; mixed: total=2, ent=1, con=0, unm=1; unclassified: total=36, ent=28, con=4, unm=4
- **ARM_A1** — static_entity_attribute: total=14, ent=10, con=1, unm=3; dynamic_action_temporal: total=31, ent=25, con=3, unm=3; mixed: total=2, ent=1, con=0, unm=1; unclassified: total=33, ent=28, con=3, unm=2
- **ARM_A7** — static_entity_attribute: total=15, ent=11, con=1, unm=3; dynamic_action_temporal: total=32, ent=23, con=3, unm=6; mixed: total=1, ent=0, con=0, unm=1; unclassified: total=35, ent=29, con=3, unm=3

## Fine-grained object / attribute / action / relation breakdown

- **ARM_A0** — object_entity: total=35, ent=27, con=3, unm=5; attribute: total=2, ent=2, con=0, unm=0; action_event: total=3, ent=3, con=0, unm=0; relation_spatial: total=0, ent=0, con=0, unm=0; mixed: total=33, ent=26, con=3, unm=4; unclassified: total=12, ent=8, con=1, unm=3
- **ARM_A1** — object_entity: total=33, ent=27, con=3, unm=3; attribute: total=3, ent=3, con=0, unm=0; action_event: total=5, ent=4, con=1, unm=0; relation_spatial: total=1, ent=1, con=0, unm=0; mixed: total=31, ent=24, con=3, unm=4; unclassified: total=7, ent=5, con=0, unm=2
- **ARM_A7** — object_entity: total=32, ent=27, con=2, unm=3; attribute: total=2, ent=2, con=0, unm=0; action_event: total=4, ent=3, con=1, unm=0; relation_spatial: total=0, ent=0, con=0, unm=0; mixed: total=35, ent=24, con=3, unm=8; unclassified: total=10, ent=7, con=1, unm=2

## Representative automatic change cases

### ARM_A0->ARM_A1
- added: `carebench_v_00009485_0` — A boy exists. [unmentioned]
- added: `carebench_v_00010336_1` — One musician is playing a violin. [contradicted]
- added: `carebench_v_00010336_1` — One musician is playing a cello. [entailed]
- replaced: `carebench_v_00007888_2` — “The woman is jumping over hurdles.” → “The woman is jumping over a hurdle.” [contradicted→contradicted]
- replaced: `carebench_v_00009795_0` — “The man is playing an instrument.” → “The man is playing an accordion.” [entailed→entailed]
- replaced: `carebench_v_00010336_1` — “There are two people.” → “There are two musicians.” [entailed→entailed]
- removed: `carebench_v_00007888_2` — The hurdles are on a track. [unmentioned]
- removed: `carebench_v_00009485_0` — A boy is present. [unmentioned]

### ARM_A1->ARM_A7
- added: `carebench_v_00002093_8` — A woman is present. [entailed]
- added: `carebench_v_00002093_8` — The makeup is black eyeliner. [unmentioned]
- added: `carebench_v_00002328_16` — A man exists. [entailed]
- replaced: `carebench_v_00002283_1` — “A woman is riding a scooter.” → “A girl is riding a scooter.” [unmentioned→entailed]
- replaced: `carebench_v_00005904_11` — “A man is present.” → “A chef is present.” [entailed→entailed]
- replaced: `carebench_v_00005904_11` — “The man is cutting vegetables.” → “The chef is cutting vegetables.” [entailed→entailed]
- removed: `carebench_v_00002093_8` — A woman exists. [entailed]
- removed: `carebench_v_00002328_16` — A man is shaving hair. [entailed]

### ARM_A0->ARM_A7
- added: `carebench_v_00002093_8` — A woman is present. [entailed]
- added: `carebench_v_00002093_8` — The makeup is black eyeliner. [unmentioned]
- added: `carebench_v_00002328_16` — A man exists. [entailed]
- replaced: `carebench_v_00002283_1` — “A woman is riding a scooter.” → “A girl is riding a scooter.” [unmentioned→entailed]
- replaced: `carebench_v_00005904_11` — “A man is present.” → “A chef is present.” [entailed→entailed]
- replaced: `carebench_v_00005904_11` — “The man is cutting vegetables.” → “The chef is cutting vegetables.” [entailed→entailed]
- removed: `carebench_v_00002093_8` — A woman exists. [entailed]
- removed: `carebench_v_00002328_16` — A man is shaving hair. [entailed]

## Focus videos

### `carebench_v_00002093_8`
- Core-event candidate (locked pre-eval): This video portrays a scene of a woman applying her makeup.
- **ARM_A0** (OK, core=1.0): a woman is applying makeup to her face  
  A woman exists. [entailed/unclassified]; The woman is applying makeup. [entailed/dynamic_action_temporal]; The makeup is being applied to the woman's face. [entailed/dynamic_action_temporal]
- **ARM_A1** (OK, core=1.0): a woman is applying makeup to her face  
  A woman exists. [entailed/unclassified]; The woman is applying makeup. [entailed/dynamic_action_temporal]; The makeup is being applied to the woman's face. [entailed/dynamic_action_temporal]
- **ARM_A7** (OK, core=1.0): a woman is applying makeup to her face with a black eyeliner  
  A woman is present. [entailed/unclassified]; The woman is applying makeup. [entailed/dynamic_action_temporal]; The makeup is being applied to the woman's face. [entailed/dynamic_action_temporal]; The makeup is black eyeliner. [unmentioned/static_entity_attribute]

### `carebench_v_00008320_0`
- Core-event candidate (locked pre-eval): The video captures the heartwarming moment of a woman embracing her dog.
- **ARM_A0** (OK, core=1.0): a woman is petting a dog  
  A woman exists. [entailed/unclassified]; The woman is petting a dog. [entailed/unclassified]; A dog exists. [entailed/unclassified]
- **ARM_A1** (OK, core=1.0): a woman is petting a dog  
  A woman exists. [entailed/unclassified]; The woman is petting a dog. [entailed/unclassified]; A dog exists. [entailed/unclassified]
- **ARM_A7** (OK, core=0.0): a woman is talking about her dog  
  A woman exists. [entailed/unclassified]; The woman is talking. [unmentioned/dynamic_action_temporal]; The woman has a dog. [entailed/unclassified]; The dog is an animal. [entailed/unclassified]

### `carebench_v_00010336_1`
- Core-event candidate (locked pre-eval): This video showcases the process of two people playing musical instruments.
- **ARM_A0** (OK, core=1.0): two people are playing instruments  
  There are two people. [entailed/static_entity_attribute]; The people are playing instruments. [entailed/dynamic_action_temporal]
- **ARM_A1** (OK, core=1.0): two musicians are playing a violin and a cello  
  There are two musicians. [entailed/static_entity_attribute]; One musician is playing a violin. [contradicted/dynamic_action_temporal]; One musician is playing a cello. [entailed/dynamic_action_temporal]
- **ARM_A7** (OK, core=1.0): two people are playing music  
  There are two people. [entailed/static_entity_attribute]; The people are playing music. [entailed/dynamic_action_temporal]

### `carebench_v_00013412_0`
- Core-event candidate (locked pre-eval): This video showcases the handshake between Mr. Donald Trump and Mr. Kim Jong-un.
- **ARM_A0** (OK, core=0.0): donald trump and barack obama are shaking hands  
  Donald Trump is a person. [entailed/static_entity_attribute]; Barack Obama is a person. [unmentioned/static_entity_attribute]; Donald Trump and Barack Obama are shaking hands. [contradicted/unclassified]
- **ARM_A1** (OK, core=1.0): the president of the united states is shaking hands with the president of north korea  
  The president of the United States is shaking hands with someone. [entailed/unclassified]; The person the president of the United States is shaking hands with is the president of North Korea. [entailed/unclassified]
- **ARM_A7** (OK, core=0.0): donald trump and barack obama are shaking hands  
  Donald Trump is a person. [entailed/static_entity_attribute]; Barack Obama is a person. [unmentioned/static_entity_attribute]; Donald Trump and Barack Obama are shaking hands. [contradicted/unclassified]

### `carebench_v_00015614_0`
- Core-event candidate (locked pre-eval): This video captures a worker actively trimming the leaves of a coconut tree.
- **ARM_A0** (OK, core=0.0): a man is operating a crane  
  A man exists. [entailed/unclassified]; The man is operating a crane. [contradicted/unclassified]
- **ARM_A1** (OK, core=0.0): a man is operating a crane  
  A man exists. [entailed/unclassified]; The man is operating a crane. [contradicted/unclassified]
- **ARM_A7** (OK, core=0.0): a crane is lifting a tree  
  A crane is lifting a tree. [contradicted/dynamic_action_temporal]

## Human annotation workload (after within-video duplicate reuse)

- Videos: **30**; caption cards in blind bundle: **90**.
- Unique captions within video: **50** (reusable duplicate cards: 40).
- Atomic claims: **248** total, **119** unique within video; 129 labels can be reused blindly within the same video.
- Estimated annotation effort: **80.9–143.1 minutes** per annotator (basis: 90-150s video watch + 30-60s core events per video; 8-15s per unique claim; 6-10s per unique caption; duplicates within a video reuse labels (blind).).

The blind bundle still covers all 90 captions and preserves arm identity only in the secret key. Duplicate reuse never changes experimental coupling or statistical weights.
