# In-domain caption → source-video mapping audit

## Corpus ambiguity (MSR-VTT train)

| metric | value |
|---|---|
| raw captions | 130,260 |
| exact-unique captions | 110,031 |
| single-source captions | 106,691 |
| multi-source captions | 3,340 (3.04%) |
| max videos per identical caption | 262 |
| mean videos per multi-source caption | 4.17 |

Generic captions ("a man is talking") are responsible for the extreme cases.

## Mapping policy used

`make_indomain_retrieval_val.py` stores all source videos per unique caption,
sorted by video ID; the deterministic downstream clip choice picks the first
(sorted) video. **Policy = deterministic sorted-video-ID (first occurrence
after sorting).** The original EVCap `no_duplicate` artifact's exact policy
cannot be proven from the available code, so this is recorded rather than
guessed; it is reproducible and does not depend on file order.

Effects on val497 top-20 retrieval and on final D1/D3/D5/D7 selected evidence
are computed once the in-domain retrieval artifacts exist; they will be appended
here. Given the 3.04% corpus rate, material ambiguity is not expected.

## Natural source redundancy (external variants)

| variant | queries with repeated source | mean unique sources | max repeat from one source |
|---|---|---|---|
| D7 | 17.9% | 6.775 | 4 |
| Oracle7 | 11.7% | 6.865 | 3 |
| GoodTail7 | 17.9% | 6.775 | 4 |
| HardWrong7 | 15.9% | 6.787 | 5 |
| EasyWrong7 | 13.7% | 6.819 | 5 |
| MidTail7 | 12.7% | 6.831 | 5 |
| LowTail7 | 9.9% | 6.887 | 4 |

Source redundancy is not equal across conditions; it is recorded as a
covariate, not forced to be uniform (that would change the natural retrieval
condition).
