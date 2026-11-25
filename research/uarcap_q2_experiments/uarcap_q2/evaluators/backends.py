"""Unified JudgeBackend adapters (Local Transformers, OpenAI-compatible HTTP, and CPU Mock)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
import re
from typing import Any
import urllib.request


@dataclass
class JudgeBackendConfig:
    backend_type: str  # local_transformers | vllm_offline | openai_compat | mock
    model_id: str
    model_family: str  # e.g., mistral | gemma | llama | gpt_oss | phi | qwen | mock
    local_model_path: str = ""
    api_base_url: str = ""
    api_key_env_var: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_new_tokens: int = 512
    timeout_sec: int = 60
    quantization: str = ""
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    enable_thinking: bool = False
    language_model_only: bool = False


class JudgeBackend(ABC):
    cfg: JudgeBackendConfig

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return raw text response from judge LLM."""

    def complete_batch(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Default batch adapter; backends with native batching should override this."""
        return [self.complete(sp, up) for sp, up in prompts]


class LocalTransformersJudgeBackend(JudgeBackend):
    """Local offline open-weight LLM judge (lazy import; runs in isolated env_text_evaluator)."""

    def __init__(self, cfg: JudgeBackendConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.local_model_path, local_files_only=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.local_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self._lazy_load()
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer([prompt_text], return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
            )
        gen = out_ids[:, inputs.input_ids.shape[1] :]
        return self._tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()


class LocalVLLMJudgeBackend(JudgeBackend):
    """Offline vLLM text judge reusing existing BN vLLM 0.29.0 + local models (e.g., Mistral Small 3.2 24B)."""

    def __init__(self, cfg: JudgeBackendConfig, vllm_engine: Any | None = None) -> None:
        self.cfg = cfg
        self._engine = vllm_engine

    def _lazy_load(self) -> Any:
        if self._engine is not None:
            return self._engine
        # FlashInfer's sampling JIT requires a `ninja` binary that is absent in this
        # shared env; the native vLLM sampler is deterministic at temperature 0.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {
            "model": self.cfg.local_model_path,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "max_model_len": self.cfg.max_model_len,
        }
        if self.cfg.quantization:
            kwargs["quantization"] = self.cfg.quantization
        if self.cfg.language_model_only:
            kwargs["language_model_only"] = True
        self._engine = LLM(**kwargs)
        return self._engine

    def complete_batch(self, prompts: list[tuple[str, str]]) -> list[str]:
        if not prompts:
            return []
        engine = self._lazy_load()
        convs = [
            [{"role": "system", "content": sp}, {"role": "user", "content": up}]
            for sp, up in prompts
        ]
        try:
            from vllm import SamplingParams  # type: ignore[import-not-found]

            sp_obj: Any = SamplingParams(temperature=self.cfg.temperature, max_tokens=self.cfg.max_new_tokens)
        except ImportError:
            sp_obj = {"temperature": self.cfg.temperature, "max_tokens": self.cfg.max_new_tokens}
        chat_kwargs: dict[str, Any] = {}
        if self.cfg.enable_thinking:
            # Only Qwen-style templates accept this flag; passing it to Gemma/Mistral is unnecessary.
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": True}
        outs = engine.chat(convs, sampling_params=sp_obj, **chat_kwargs)
        res: list[str] = []
        for o in outs:
            txt = str(o.get("text", "")) if isinstance(o, dict) else str(o.outputs[0].text)
            res.append(txt.strip())
        return res

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete_batch([(system_prompt, user_prompt)])[0]


class SubprocessTransformersJudgeBackend(JudgeBackend):
    """Cross-env local LLM judge backend invoking `judge_worker.py` in isolated `uarcap_q2_evaluator` env."""

    def __init__(
        self,
        cfg: JudgeBackendConfig,
        python_executable: str | None = None,
        conda_env: str | None = "uarcap_q2_evaluator",
        gpu_lock_path: Any | None = None,
        gpu_uuid: str | None = None,
        require_nvidia_smi_idle: bool = False,
        subprocess_runner: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.python_executable = python_executable
        self.conda_env = conda_env
        self.gpu_lock_path = gpu_lock_path
        self.gpu_uuid = gpu_uuid
        self.require_nvidia_smi_idle = require_nvidia_smi_idle
        self.subprocess_runner = subprocess_runner

    def complete_batch(self, prompts: list[tuple[str, str]]) -> list[str]:
        from dataclasses import asdict
        from pathlib import Path
        import tempfile
        from ..common.gpu_lock import GPULockConfig
        from ..common.gpu_runtime import run_isolated_gpu_subprocess
        from ..common.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl

        if not prompts:
            return []
        worker_script = Path(__file__).resolve().parent / "judge_worker.py"
        lock_cfg = (
            GPULockConfig(
                lock_path=Path(self.gpu_lock_path),
                run_id="text_judge_batch",
                gpu_uuid=self.gpu_uuid,
                require_nvidia_smi_idle=self.require_nvidia_smi_idle,
            )
            if self.gpu_lock_path
            else None
        )
        with tempfile.TemporaryDirectory(prefix="uarcap_judge_") as td:
            td_p = Path(td)
            cfg_p = td_p / "judge_cfg.json"
            req_p = td_p / "requests.jsonl"
            resp_p = td_p / "responses.jsonl"
            atomic_write_json(cfg_p, asdict(self.cfg))
            atomic_write_jsonl(
                req_p,
                [
                    {"request_id": f"r{i:06d}", "system_prompt": sp, "user_prompt": up}
                    for i, (sp, up) in enumerate(prompts)
                ],
            )
            proc = run_isolated_gpu_subprocess(
                script_or_module=str(worker_script),
                script_args=["--config-json", str(cfg_p), "--requests-jsonl", str(req_p), "--responses-jsonl", str(resp_p)],
                python_executable=self.python_executable,
                conda_env=self.conda_env,
                gpu_lock_cfg=lock_cfg,
                timeout_sec=max(300, self.cfg.timeout_sec * len(prompts)),
                subprocess_runner=self.subprocess_runner,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Text judge worker failed (rc={proc.returncode}): {proc.stderr[-1000:]}")
            if not resp_p.is_file():
                raise RuntimeError(f"Text judge worker produced no responses JSONL: {resp_p}")
            by_id = {r["request_id"]: str(r["response"]) for r in read_jsonl(resp_p)}
            return [by_id[f"r{i:06d}"] for i in range(len(prompts))]

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete_batch([(system_prompt, user_prompt)])[0]


class OpenAICompatJudgeBackend(JudgeBackend):
    """Works with local vLLM/SGLang OpenAI-compatible server OR optional API sensitivity check."""

    def __init__(self, cfg: JudgeBackendConfig) -> None:
        self.cfg = cfg

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self.cfg.api_base_url:
            raise ValueError("OpenAICompatJudgeBackend requires api_base_url")
        url = self.cfg.api_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.cfg.api_key_env_var, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = json.dumps(
            {
                "model": self.cfg.model_id,
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_new_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"]).strip()


class MockJudgeBackend(JudgeBackend):
    """Deterministic CPU mock judge supporting clean parsing and explicit fault injection."""

    def __init__(
        self,
        model_id: str = "mock_crossfamily_judge_cpu",
        model_family: str = "mock_mistral",
        fault_mode: str | None = None,
    ) -> None:
        self.cfg = JudgeBackendConfig(backend_type="mock", model_id=model_id, model_family=model_family)
        self.fault_mode = fault_mode

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.fault_mode == "invalid_json":
            return "{not valid json!!!"
        if self.fault_mode == "empty":
            return ""
        if self.fault_mode == "refusal":
            return "As an AI language model, I cannot evaluate this video description."
        if self.fault_mode == "truncated":
            return '{"claims": [{"claim": "a man walks", "label": "entailed"'
        if self.fault_mode == "illegal_label":
            return json.dumps({"claims": [{"claim": "x", "label": "partially_true", "evidence": "maybe"}]})
        if self.fault_mode == "missing_field":
            return json.dumps({"claims": [{"claim": "x"}]})
        if self.fault_mode == "no_evidence":
            return json.dumps({"claims": [{"claim": "x", "label": "contradicted", "evidence": ""}]})
        if self.fault_mode == "mismatched_claim":
            return json.dumps({"claims": [{"claim": "TOTALLY DIFFERENT UNRELATED CLAIM", "label": "entailed", "evidence": "ref"}]})
        if self.fault_mode == "duplicate_claims":
            return json.dumps(
                {
                    "claims": [
                        {"claim": "a dog runs", "label": "entailed", "evidence": "ref"},
                        {"claim": "a dog runs", "label": "entailed", "evidence": "ref"},
                    ]
                }
            )

        # Dispatch by prompt task marker
        if "[TASK:EXTRACT_CLAIMS]" in user_prompt:
            m = re.search(r"CAPTION:\s*(.+)", user_prompt, re.S)
            cap = m.group(1).strip() if m else ""
            # Split cleanly on ' and ' or punctuation into atomic clauses
            raw_parts = [p.strip(" .") for p in re.split(r"[.;]|,\s+and\s+|\s+and\s+", cap) if p.strip(" .")]
            claims = [p for p in raw_parts if len(p.split()) >= 2] or ([cap.strip(" .")] if cap.strip() else [])
            return json.dumps({"claims": claims}, ensure_ascii=False)

        if "[TASK:CORE_EVENT_CHECK_SINGLE]" in user_prompt:
            m_cap = re.search(r"CAPTION:\s*(.*?)\nCORE_EVENT:", user_prompt, re.S)
            m_ev = re.search(r"CORE_EVENT:\s*(.+)", user_prompt, re.S)
            cap = (m_cap.group(1).strip().lower() if m_cap else "")
            event = (m_ev.group(1).strip().lower() if m_ev else "")
            cap_words = set(re.findall(r"[a-z0-9']+", cap))
            e_words = set(re.findall(r"[a-z0-9']+", event))
            overlap = len(e_words & cap_words) / max(len(e_words), 1)
            if overlap >= 0.45:
                return json.dumps({"label": "entailed", "evidence": "Captured by the caption."})
            return json.dumps({"label": "unmentioned", "evidence": "Caption omits this event."})

        if "[TASK:CORE_EVENT_CHECK]" in user_prompt:
            m_cap = re.search(r"CAPTION:\s*(.*?)\nCORE_EVENTS_JSON:", user_prompt, re.S)
            m_ev = re.search(r"CORE_EVENTS_JSON:\s*(\[.*\])", user_prompt, re.S)
            cap = (m_cap.group(1).strip().lower() if m_cap else "")
            cap_words = set(re.findall(r"[a-z0-9']+", cap))
            events = json.loads(m_ev.group(1)) if m_ev else []
            verdicts = []
            for event in events:
                e_words = set(re.findall(r"[a-z0-9']+", str(event).lower()))
                overlap = len(e_words & cap_words) / max(len(e_words), 1)
                if overlap >= 0.45:
                    verdicts.append({"label": "entailed", "evidence": "Captured by the caption."})
                else:
                    verdicts.append({"label": "unmentioned", "evidence": "Caption omits this event."})
            return json.dumps({"verdicts": verdicts}, ensure_ascii=False)

        if "[TASK:3WAY_REF_NLI]" in user_prompt:
            m_ref = re.search(r"REFERENCE:\s*(.*?)\nCLAIMS_JSON:", user_prompt, re.S)
            m_clm = re.search(r"CLAIMS_JSON:\s*(\[.*\])", user_prompt, re.S)
            ref = (m_ref.group(1).strip().lower() if m_ref else "")
            ref_words = set(re.findall(r"[a-z0-9']+", ref))
            claims = json.loads(m_clm.group(1)) if m_clm else []
            out_items = []
            for cl in claims:
                c_low = cl.lower()
                c_words = set(re.findall(r"[a-z0-9']+", c_low))
                overlap = len(c_words & ref_words) / max(len(c_words), 1)
                if "raking leaves" in c_low and "throwing grass" in ref:
                    lbl, ev = "contradicted", "Reference states throwing grass, conflicting with raking leaves."
                elif "near a " in c_low and overlap < 0.75:
                    lbl, ev = "unmentioned", "Added tail entity not mentioned in reference."
                elif overlap >= 0.45:
                    lbl, ev = "entailed", "Supported by overlapping events in reference."
                else:
                    lbl, ev = "unmentioned", "Not described in the reference description."
                out_items.append({"claim": cl, "label": lbl, "evidence": ev})
            return json.dumps({"claims": out_items}, ensure_ascii=False)

        if "[TASK:E3_SINGLE_TARGET]" in user_prompt:
            m_cap = re.search(r"CAPTION:\s*(.*?)\nTARGET_PROPOSITION:", user_prompt, re.S)
            m_tgt = re.search(r"TARGET_PROPOSITION:\s*(.+)", user_prompt, re.S)
            cap = (m_cap.group(1).strip() if m_cap else "")
            tgt = (m_tgt.group(1).strip() if m_tgt else "")
            cap_low, tgt_low = cap.lower(), tgt.lower()
            neg_pat = re.compile(rf"\b(not|no|never|rather than|instead of)\b[^.]*{re.escape(tgt_low[:12])}")
            negated = bool(neg_pat.search(cap_low))
            hedged = bool(re.search(r"\b(maybe|might|possibly|appears to|either|or)\b", cap_low))
            generic_only = bool(re.search(r"\b(doing something|working in the yard|doing chores)\b", cap_low)) and (
                tgt_low not in cap_low
            )
            tgt_words = [w for w in re.findall(r"[a-z0-9']+", tgt_low) if len(w) > 2]
            cap_words = set(re.findall(r"[a-z0-9']+", cap_low))
            hit_ratio = sum(1 for w in tgt_words if w in cap_words) / max(len(tgt_words), 1)
            expressed = (hit_ratio >= 0.6) and not negated and not generic_only
            return json.dumps(
                {
                    "expressed": bool(expressed),
                    "negated": bool(negated),
                    "hedged": bool(hedged and expressed),
                    "generic_only": bool(generic_only),
                    "evidence_span": cap[:80] if (expressed or negated) else "",
                },
                ensure_ascii=False,
            )

        raise ValueError("Unknown mock judge prompt task")
