set -e

# if no uv, install it
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env 
fi

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .[math,geo,vllm]
uv pip install flash-attn --no-build-isolation
uv pip install -e .[gpu]
uv pip install pip
bash scripts/uvinstall_deepeyes.sh
uv pip install tensorboard
