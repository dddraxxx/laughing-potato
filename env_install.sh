uv pip install -e .[math,geo,vllm]
uv pip install flash-attn --no-build-isolation
uv pip install -e .[gpu]
bash scripts/install_deepeyes.sh