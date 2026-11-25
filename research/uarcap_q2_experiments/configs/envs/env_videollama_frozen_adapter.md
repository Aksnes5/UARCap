# Isolated Environment 1: Frozen Video-LLaMA / EVCap Adapter (READ-ONLY)

**CRITICAL RULE:** NEVER run `pip install`, `conda update`, or upgrade `transformers`/`torch` inside this historical environment!

- **Interpreter:** `/media/bn/37b77782-9321-42a0-8a20-f53b4ec9179d1/xyl/Display/repro_evcap/venv/bin/python`
- **Base Conda:** `/home/bn/anaconda3/envs/TTP` (`Python 3.9.13`, `torch 2.1.2+cu121`)
- **Pinned Packages:** `transformers==4.28.0`, `timm==0.6.13`
- **Usage:** Invoked strictly as a read-only subprocess by `uarcap_q2/generators/videollama_adapter.py` using frozen `eval_evcap_fix_video_llama_xclip_msrvtt_fusion_gate_prompt5.py`.
