set -e

uv pip install -e .[math,geo,vllm] --system
uv pip install flash-attn --no-build-isolation --system
uv pip install -e .[gpu] --system
uv pip install pip --system
uv pip install tensorboard --system

uv pip install evaluate --system
uv pip install transformers==4.51.3 --system
uv pip install vllm==0.8.2 --system
uv pip install -U pynvml --system
uv pip install mathruler --system
uv pip install pydantic --upgrade --system
uv pip install openai --upgrade --system
uv pip install tensordict==0.6.2 --system
uv pip install triton==3.2.0 --system
uv pip install qwen_vl_utils --system
uv pip install math_verify --system