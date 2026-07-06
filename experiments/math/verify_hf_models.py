#!/usr/bin/env python3
"""Verify that configured Hugging Face model ids are accessible before long runs."""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
from importlib import metadata
from pathlib import Path
from typing import Any

from huggingface_hub import model_info
from packaging.version import Version
from transformers import AutoConfig

from verl.utils.model import get_hf_auto_model_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="Hugging Face model ids to verify.")
    parser.add_argument(
        "--allow-automodel-fallback",
        action="store_true",
        help="Allow generic AutoModel fallback. By default this is rejected because RL generation expects a task head.",
    )
    parser.add_argument(
        "--load-smoke-model",
        help="Optionally load one model with the same Transformers AutoModel class selected by the FSDP worker.",
    )
    parser.add_argument(
        "--vllm-smoke-model",
        help="Optionally instantiate vLLM for one model and generate a tiny response.",
    )
    parser.add_argument("--vllm-tensor-parallel-size", default=1, type=int)
    parser.add_argument("--vllm-max-model-len", default=1024, type=int)
    parser.add_argument("--vllm-gpu-memory-utilization", default=0.70, type=float)
    parser.add_argument("--vllm-enforce-eager", dest="vllm_enforce_eager", action="store_true", default=True)
    parser.add_argument("--vllm-cuda-graphs", dest="vllm_enforce_eager", action="store_false")
    return parser.parse_args()


def resolve_config_source(model_id: str) -> str:
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        print("local_model_ok:", {"path": str(model_path)})
        return str(model_path)

    info = model_info(model_id)
    if getattr(info, "private", False):
        raise SystemExit(f"model is private: {model_id}")
    if getattr(info, "gated", False):
        raise SystemExit(f"model is gated: {model_id}")
    if getattr(info, "disabled", False):
        raise SystemExit(f"model is disabled: {model_id}")
    print(
        "hf_model_ok:",
        {
            "id": info.modelId,
            "sha": getattr(info, "sha", None),
            "pipeline_tag": getattr(info, "pipeline_tag", None),
        },
    )
    return model_id


def verify_model_config(model_id: str, *, allow_automodel_fallback: bool) -> tuple[str, type[Any]]:
    config_source = resolve_config_source(model_id)
    try:
        config = AutoConfig.from_pretrained(config_source, trust_remote_code=True)
    except Exception as exc:
        raise SystemExit(
            f"transformers_model_config_failed: {model_id}\n"
            f"{type(exc).__name__}: {exc}\n"
            "This model exists on Hugging Face, but the installed Transformers stack cannot load "
            "its architecture. Install a Transformers build that supports this model before running this phase."
        ) from exc
    print("transformers_config_ok:", {"id": model_id, "model_type": getattr(config, "model_type", None)})
    auto_model_cls = get_hf_auto_model_class(config)
    auto_class = auto_model_cls.__name__
    if auto_class in {"unsupported", "AutoModel"} and not allow_automodel_fallback:
        raise SystemExit(
            f"verl_model_class_unsupported: {model_id}\n"
            f"selected_auto_class={auto_class}, architectures={getattr(config, 'architectures', None)}\n"
            "The installed Transformers stack can read the config, but it does not expose a language-generation "
            "AutoModel class that the FSDP worker can use safely. Upgrade Transformers/vLLM or choose another "
            "text-generation checkpoint."
        )
    print("verl_auto_class_ok:", {"id": model_id, "auto_class": auto_class})
    return config_source, auto_model_cls


def load_transformers_smoke(model_id: str, auto_model_cls: type[Any]) -> None:
    import torch

    base_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if torch.cuda.is_available():
        base_kwargs["device_map"] = "auto"

    attempts = [
        {**base_kwargs, "dtype": "auto"},
        {**base_kwargs, "torch_dtype": "auto"},
        {k: v for k, v in {**base_kwargs, "dtype": "auto"}.items() if k != "attn_implementation"},
        {k: v for k, v in {**base_kwargs, "torch_dtype": "auto"}.items() if k != "attn_implementation"},
    ]

    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            model = auto_model_cls.from_pretrained(model_id, **kwargs)
            first_param = next(model.parameters(), None)
            print(
                "transformers_load_smoke_ok:",
                {
                    "id": model_id,
                    "auto_class": auto_model_cls.__name__,
                    "device": str(first_param.device) if first_param is not None else "unknown",
                    "dtype": str(first_param.dtype) if first_param is not None else "unknown",
                },
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        except Exception as exc:
            last_exc = exc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise SystemExit(
        f"transformers_load_smoke_failed: {model_id}\n"
        f"{type(last_exc).__name__}: {last_exc}"
    )


def load_vllm_smoke(
    model_id: str,
    *,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    enforce_eager: bool,
) -> None:
    try:
        numpy_version = metadata.version("numpy")
        numba_version = metadata.version("numba")
    except metadata.PackageNotFoundError:
        numpy_version = ""
        numba_version = ""
    if numpy_version and Version(numpy_version) >= Version("2.3"):
        raise SystemExit(
            "vllm_numpy_numba_incompatible:\n"
            f"numpy={numpy_version}, numba={numba_version or 'not_installed'}\n"
            "vLLM imports numba in this stack, and numba requires NumPy 2.2 or less. "
            'Run: uv pip install -q -U "numpy==2.1.0"'
        )

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise SystemExit(f"vllm_import_failed: {type(exc).__name__}: {exc}") from exc

    try:
        vllm_version = metadata.version("vllm")
    except metadata.PackageNotFoundError:
        vllm_version = "unknown"
    print("vllm_version:", vllm_version)

    def run_llm_smoke(utilization: float) -> None:
        llm = LLM(
            model=model_id,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=utilization,
            dtype="auto",
            enforce_eager=enforce_eager,
        )
        outputs = llm.generate(["What is 1+1? Answer briefly."], SamplingParams(max_tokens=8, temperature=0.01))
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        print(
            "vllm_load_smoke_ok:",
            {
                "id": model_id,
                "tensor_parallel_size": tensor_parallel_size,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": utilization,
                "enforce_eager": enforce_eager,
                "sample": text.strip(),
            },
        )
        del llm
        gc.collect()

    try:
        run_llm_smoke(gpu_memory_utilization)
    except Exception as exc:
        raise SystemExit(f"vllm_load_smoke_failed: {model_id}\n{type(exc).__name__}: {exc}") from exc


def main() -> None:
    args = parse_args()
    verified: dict[str, tuple[str, type[Any]]] = {}
    for model_id in dict.fromkeys(args.models):
        verified[model_id] = verify_model_config(model_id, allow_automodel_fallback=args.allow_automodel_fallback)

    if args.load_smoke_model:
        _, auto_model_cls = verified.get(args.load_smoke_model) or verify_model_config(
            args.load_smoke_model, allow_automodel_fallback=args.allow_automodel_fallback
        )
        load_transformers_smoke(args.load_smoke_model, auto_model_cls)

    if args.vllm_smoke_model:
        load_vllm_smoke(
            args.vllm_smoke_model,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            enforce_eager=args.vllm_enforce_eager,
        )


if __name__ == "__main__":
    main()
