# Resource and runtime plan

The frozen project uses one NVIDIA RTX A6000 48 GB-class inference device for this captioning setting, batch size 1, 16 frames, Vicuna-7B-v1.3, beam width 5 and a frozen checkpoint. Use the existing runtime, model/tokenizer cache, frame TSV and external feature cache. Neither follow-up requires retraining, extracting new features, building a new index, or starting a service. No current GPU availability or measured new-arm VRAM peak is known.

| Source | Recorded runtime and planning use |
| --- | --- |
| Frozen `reports/runtime_ledger.csv`, `test2990_D7` | 8,521 generation seconds for 2,990 queries, about 2.37 GPU-hours per full-sequence arm as an anchor. |
| Same ledger, nonzero 497-query entries | 834–2,777 generation seconds, about 1.68–5.59 s/query. Zero-time completion placeholders are excluded. Different configurations and the earlier validation loader make this a planning envelope. |
| Full new A or B matrix | Three full-order arms, 8,970 query decisions. Historical throughput implies about 4.2–13.9 GPU-hours per experiment before trace and startup overhead, with the direct test anchor about 7.1 GPU-hours. |
| Both matrices | Six arms, 17,940 generated query decisions, about 8.4–27.9 GPU-hours at the historical envelope. Primary evaluation uses 2,690 A and 2,618 B queries. The remaining generated queries preserve archived order. |

Exact seconds depend on caption length, beam termination, runtime I/O, tensor hashes copied to CPU and initialization. B hashes three cached `[32,768]` tensors per query plus the visual representations and query frames. Keep the device single-job during future timing and obtain an actual timed smoke plus peak memory before reserving the full allocation. The frozen validation ledger and test anchor predate this adapter, so neither is a measured follow-up runtime. Optional A sensitivity would add a separate full arm and is outside these estimates.

Existing deployment data needed: `evcap_external/test2990/test2990_16frames.yaml` and its image/caption/label TSV paths and offsets, the pinned MSR-VTT checkpoint named by the archived test runner, the existing Vicuna and base encoder caches, and actual representative `.pt` tensors in the faithful external feature cache. Derive the exact clip list for A or B from the named CAP inputs. B needs only the three head clips per query. Local CPU preparation uses the frozen D cap/sims, 13.1 MB raw top20 artifact, and two small query manifests. Treat preflight data I/O, driver and model availability as separately receipted deployment conditions.
