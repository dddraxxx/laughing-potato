set -e
# if no uv, install it
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

uvp(){
    uv pip install $@ --system
}
uvp -e .[math,geo,vllm]
uvp flash-attn --no-build-isolation
uvp -e .[gpu]
uvp pip
uvp tensorboard
uvp nvitop

uvp evaluate
uvp transformers==4.51.3
uvp vllm==0.8.2
uvp -U pynvml
uvp mathruler
uvp pydantic --upgrade
uvp openai --upgrade
uvp tensordict==0.6.2
uvp triton==3.2.0
uvp qwen_vl_utils
uvp math_verify

uvp ipykernel jupyterlab matplotlib