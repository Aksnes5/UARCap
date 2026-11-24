# Experiment A implementation readiness

**Conditional GO.** The matched k=0 behavior passes Windows source and CPU checks. The actual checkpoint, tensors, dtype under CUDA/autocast, weights, tokenizer and generation remain to be checked in authorized smoke.

| Check | Local finding |
| --- | --- |
| Parameter parsing | CPU test calls the actual archived parser and constructor expressions. All three A0 flags parse as 0. The evaluator converts inner zeros to `None`; EVCap falls back to outer `topn=0` for each. A1/A7 resolve to (1,1) and (7,7). Keep all three zero flags together for A0. |
| Text and cache | Every A0 CAP is `[[],[],[]]` and every SIM is `[[],[]]`. The real frozen method bodies, exercised with CPU tensors, used zero captions and made zero cache calls. Trace and artifact verification enforce this per query. |
| Query and prompt | Retain `use_fusion=True`, `retriever=True`, `no_retrieval=False`, both `<VideoHere>` slots, and the query encoder/LLM projections. The frozen prompt wrapper uses its normal text placeholder with the empty-evidence newline. The CPU path verifies both slots, finite embeddings, batch-one length and int32 all-one attention mask. |
| Attention and gating | `fusion_train` emits a finite zero float32 aggregate `[1,32,768]` in the tested default dtype. The aggregate feeds both K and V. Query hidden states and learned projections, biases and gating still operate. The float32 fallback needs checking against the real device and autocast path. The CPU example also confirms that a real attention/gating path with zero K/V can produce a nonzero fused projection. |
| Arms held constant | Frame tensor, both Q-Former input groups, query hidden and query projection, encoder module modes and CPU/CUDA RNG state, visual mask, references and prompt skeleton are checked across A1, A7, A0. A's aggregate, gates and final wrapped prompt can differ with the evidence intervention. |

`runtime_trace.py` extends the frozen hook without changing the parser's semantics. It records actual cache load calls, query inputs, K/V and aggregate signatures, masks, capture for the existing gates, and prompt signatures. On the frozen torch architecture required by the hook, any unexpected shape/load or non-finite observation stops execution. The overlay also replaces the trace-side `zip(strict=True)` call with an explicit length check for archive Python 3.9 compatibility.

Run the approved smoke with A0/A1/A7 on the first eight test IDs and verify the A family bundle before attempting any A full arm. On dtype, mask, fallback or query parity failure, preserve logs and block A. A0 tests the matched checkpoint with zero external evidence in its trained fusion path. It does not model removal of the inherited fusion architecture or retraining with zero evidence.

The inherited evaluator leaves the parent EVCap mode as constructed and calls `.eval()` on the LLM within beam search. The video Q-Former freeze flag disables gradients. It does not switch its module mode, and dropout is possible in the inherited encoding path. Preserve these semantics and verify actual mode, per-query RNG and query tensor parity. The CPU fixture exercises active query-side dropout with aligned seeds. There is no proof yet that the BN run will remain aligned.
