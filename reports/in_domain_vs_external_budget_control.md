# Matched in-domain vs external context-budget control (val497)

Same queries, query features, checkpoint, prompt, decoding, fusion and
FAITHFUL_V2 candidate features; only the retrieval corpus changes.

| k | External FULL (D) | In-domain train (D) |
|---|---|---|
| 1 | 51.77 | 54.16 |
| 3 | 53.14 | 55.14 |
| 5 | 50.51 | 55.00 |
| 7 | 50.05 | **55.43** |

In-domain paired effects: D3−D7 = −0.27 (CI [−2.30,+1.73], p=0.80),
D1−D7 = −1.24 (p=0.25), D1−D3 = −0.97 (p=0.18), D3−D5 = +0.17 (p=0.83):
the in-domain curve is flat and statistically indistinguishable across k=1–7.

External FULL shows the opposite: D3−D7 = +3.11 (CI [+0.72,+5.70], p=0.012)
on val497, and the same direction on untouched test.

## Pattern
`EXTERNAL_EVIDENCE_OVERLOAD`

The same frozen generator can absorb seven in-domain contexts without loss,
while seven external contexts degrade generation by ~3 CIDEr relative to the
best external budget. Large-k degradation is therefore not a generic inability
of the generator to use many contexts; it is specific to external evidence in
this setting.

Caveat: in-domain retrieval corpus is much smaller (MSR-VTT train, 6,513 videos)
than VideoUFO FULL (1.09 M clips), so corpus size and evidence source are
confounded by design in this control; the result locates the effect to the
external-evidence condition rather than isolating corpus size alone.
