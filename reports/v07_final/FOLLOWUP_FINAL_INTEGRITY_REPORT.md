# Final integrity and scoring report

**PASS.** Windows replay validated A full, B smoke and B full before either outcome was scored. The replayed gate objects matched their delivered counterparts in every field. The accepted A smoke gate was bound separately to the earlier evidence. Full output comprised 17,940 predictions and 17,940 trace rows in six arms. Every full arm has exactly one start, one successful end, one first-attempt receipt and a valid completion seal. The 24 B smoke generations are additional and excluded from these totals and statistics.

## Frozen material and checks

| Component | Verification |
| --- | --- |
| GPU delivery | Fetched `8ec143b9dd0a3b6f2a03fdd95b3d8ce3016230c0` from the named remote branch. Checked manifest identity against the Git object, then content size and SHA-256 for all 45 entries, including ZIPs, configs, runner events, runtime summaries and smoke evidence. |
| Seals and local rebinding | For each ZIP, compared byte size, digest and member list with the delivery witnesses. Read predictions, JSONL and receipt at the declared relative paths. Bound their filenames to the Linux scorer manifest and verified bytes against the ZIP seal, runtime summary and full scorer manifest. The Windows manifests reference the extracted originals. |
| A smoke | The accepted gate is identical to both archived BN validation objects. All nine underlying prediction, trace and run-receipt hashes agree with their original Git blobs and eight-row completion seals. No new inference or smoke revalidation was required. |
| A full | Per arm, 2,990 distinct ordered queries with nonempty predictions, correct references, and 2,990 aligned traces. A0 supplied zero captions, loaded zero features and had no visual softmax. Its 1 × 32 × 768 external aggregate and attention K/V matched the zero tensor. Frames, Q-Former inputs, query projection, masks, module state and CPU/CUDA encoder RNG fingerprints satisfied cross-arm parity. A1 and A7 used their archived D prefixes. |
| B smoke and full | Eight and 2,990 queries per arm respectively. Identical query frames, representative IDs, raw scores, ordered feature-tensor hashes, weights, aggregate, attention K/V, gate inputs, visual projections and mask, module state, and encoder RNG across the three arms. Observed text hashes and counts agreed with frozen arm inputs. For the 372 full queries outside the eligible set, both extensions also matched B3's full prediction and trace records exactly. |
| Checkpoint and runtime | All full-arm receipts bind the archived evaluator hashes and fixed engine arguments. Common inputs match the delivered configurations, including checkpoint, frame witnesses, LLM and environment manifests. The checkpoint SHA-256 is `a8b5af79d55a2c7e684771475d645be6373208bdaae5a4211bee556581eb8002`. Runtime configuration is attested by receipts. Windows independently read delivered bytes and traces. Model weights and videos stayed on the original host. |
| Module state | Throughout all six arms, the parent, video Q-Former and attention were in training mode. The image Q-Former and LLM were in evaluation mode. The inherited mode was preserved. Equal encoder RNG and intermediate tensors are relevant evidence for the paired comparisons. |
| Cohorts | Full 2,990 and pilot-excluded 2,690 manifests passed byte hashes. B contains 2,618 unique eligible queries, byte SHA `c17126c4b6924b1eeb2bdf6b8021d8ebf2290966dbe14305def25cf25e1390e4`, canonical SHA `450770ffd92520a8b474186c75c9e08cf7fbe3d11949f227a987e793cef03972`. Reference lists, including order within query, matched the archived witness. |
| Historical endpoints | All 2,990 parsed output records agree for each of A1 vs archived D1 and A7 vs archived D7. Historical processes lack the new runtime traces. Matching records establish endpoint reproducibility on the same clips. |
| Formal evaluation | Unchanged `evaluate_followup.py` from `ee7d3b9876dea94a1ee9eeb2626b92102c85d458` and audit statistics from `d80b0876c8e8ef403e7528be02f75a56c378d40d`. Deployed validator matched `94f5529f8a6e4424d2af207df42482cf87c7ce1b`. The scorer read the pinned COCO CIDEr files, Stanford PTB jar and hash-matched source metadata. Official inference used only the PTB arrays. |

## Run record

Python 3.11.9, NumPy 2.4.6, Microsoft OpenJDK 17.0.20.1 and Matplotlib 3.11.2 were used locally. The PTB jar SHA-256 is `2fcb91bb7a111f93d71e264f4ee0e3afd19ba0dde6d21b38605088df9e940399`. The source metadata mirror matched the independent audit SHA-256 `3540d91f9fd5f9a56d75b00c6576f30814fd19395d027791a6d438e9dc297014`. All source labels were resolved, yielding 2,004 clusters in A and 1,969 in B. Each family used the same declared seed, 20261125, with 10,000 bootstrap resamples and 10,000 two-sided sign flips. [Metadata mirror](https://huggingface.co/datasets/friedrichor/MSR-VTT/blob/main/raw_data/MSRVTT_data.json)

The local input directory came from the previously verified Windows handoff. Its transferable copy is under `research/uarcap_v06_followup_deployment/local_cpu_handoff/transfer` at `9f8f8ca5c10612503fadad6359ac9129968d399e`. Use the frozen ZIP bytes and a checkout preserving LF (on Windows, clone with `git -c core.autocrlf=false clone --config core.autocrlf=false …`). Use a fresh replay work directory or reuse sealed input bytes in a work directory without prior gate/scoring outputs. The frozen tools create new result files exclusively. The source metadata, PTB jar and a prepared directory with the eight named JSONs are input arguments. The replay reads the small archived inputs and reference witness from Git and creates paths for the new machine. No access to BN or GPU is required. [Frozen input handoff](https://github.com/Aksnes5/UARCap/tree/9f8f8ca5c10612503fadad6359ac9129968d399e/research/uarcap_v06_followup_deployment/local_cpu_handoff)

From the repository root, the Windows run used the following commands. The first generated fresh formal results. The other two render displays and check released numbers against the scoring artifacts.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:PYTHONUTF8='1'
python -B reports/v07_final/replay.py --work D:/UARCap/_v07_local --prepared D:/UARCap/uarcap_v06_prepared_inputs_local --metadata D:/UARCap/_v07_local/MSRVTT_data.json --jar D:/EVCapCode/evaluation/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar --java D:/12306/.tools/jdk17/bin/java.exe --out reports/v07_final
python -B paper/v0.7_assets/build_figures.py
python -B reports/v07_final/check_release.py --work D:/UARCap/_v07_local
```

The command receipt retains all validator and evaluator arguments, raw asset and implementation digests, completion-seal hashes and gate evidence. The release check records output identity and figure digests. Hardware identifiers and the machine account are omitted from the public command receipt. [Integrity and commands](INTEGRITY_AND_COMMANDS.json) · [Release check](RELEASE_CHECK.json) · [Windows manifest A](SCORER_MANIFEST_A_WINDOWS.json) · [Windows manifest B](SCORER_MANIFEST_B_WINDOWS.json)

## Sources for the manuscript

| Display or claim | Numerical source |
| --- | --- |
| Fig. 1 and study variables | Frozen arms in `launch_pinned.py`, corpus manifests in the independent audit, B construction in `prepare_inputs.py` |
| Fig. 2 and Table 1 | Historical audit `results/top1_churn_downstream.csv`, `results/paired_contrasts.csv` and `results/summary.json`. Candidate-only endpoint scores use the D7 cells of `reports/val_retrieval_grid_10k.json` and `reports/val_retrieval_grid.json` at the historical base commit. |
| Fig. 3 and Tables 2–3 | Historical audit `summary.json` official PTB means and `paired_contrasts.csv` benchmark families |
| Figs. 4–5 and Tables 4–5 | Both formal JSONs and the five-test Holm CSV below. B matching statistics were independently checked against the consumed quartet texts and scores for all eligible queries. Prompt lengths come directly from eligible runtime trace rows. |
| Table 6 and Appendix A | Historical audit `paired_contrasts.csv` Oracle, repaired mechanism and leave-one-out families, with `refit_idf_sensitivity.json` for refitted IDF |
| Published predecessor | Fang et al., §§3.4–3.5 and 4.1–4.3, Tables 1–4. Locally inspected eight-page PDF SHA-256 `0ac76d0732a7c37fa730137176c384d844c395c0a5f49bc9da979e52334e80ac`. [Published article](https://doi.org/10.1016/j.displa.2026.103420) |

[A formal results](FOLLOWUP_A_FORMAL_RESULTS.json) · [B formal results](FOLLOWUP_B_FORMAL_RESULTS.json) · [Family tests](FOLLOWUP_PRIMARY_CONTRASTS.csv) · [All five tests](FOLLOWUP_FIVE_TEST_HOLM_RESULTS.csv) · [Historical audit numbers](../local_independent_review/results) · [Figure code](../../paper/v0.7_assets/build_figures.py)
