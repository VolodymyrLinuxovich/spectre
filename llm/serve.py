"""Self-hosted fallback LLM server: vLLM OpenAI-compatible API on Modal.

Primary path is a Modal Dedicated Endpoint (see llm/endpoint.sh). This app
(`spectre-llm-vllm`) is the backup if the managed endpoint is unavailable.

Deploy:  modal deploy llm/serve.py
URL:     printed by deploy; OpenAI base_url is <url>/v1
"""

import subprocess

import modal

MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
SERVED_NAME = "spectre-coder"
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

hf_cache = modal.Volume.from_name("spectre-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("spectre-vllm-cache", create_if_missing=True)

app = modal.App("spectre-llm-vllm")


@app.function(
    image=image,
    gpu="H100",
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    timeout=60 * 60,
    scaledown_window=60 * 60,  # keep warm through the demo
    min_containers=0,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=20 * 60)
def serve():
    cmd = [
        "vllm", "serve", MODEL,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--max-model-len", "65536",
        "--gpu-memory-utilization", "0.92",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
    ]
    subprocess.Popen(cmd)
