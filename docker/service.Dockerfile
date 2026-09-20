# The Laya triage service as a Linux container, for the kind deployment.
#
#   docker build -f docker/service.Dockerfile -t laya-router:cpu .
#
# Weights are NOT baked in. The 843 MB checkpoint is mounted from the host's Hugging Face cache, so the image
# stays small and a checkpoint change does not mean an image rebuild.
#
# This is a CPU image on purpose. A Linux container on macOS cannot reach Apple's MPS backend, so a containerised
# run on this laptop falls back to CPU and is slower than `laya-router serve` run natively. docs/EVAL.md reports
# both numbers rather than quietly using the faster one.
FROM python:3.12-slim

ARG TORCH_VERSION=2.9.1
ENV PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /usr/local/bin/uv

WORKDIR /app

# Dependency layer first, so editing the source does not reinstall PyTorch.
COPY pyproject.toml uv.lock README.md ./
# uv.lock pins torch from PyPI, whose Linux wheels drag in several GB of CUDA libraries this image never uses.
# Everything else is installed exactly as locked; torch alone comes from the CPU index.
RUN uv export --locked --no-emit-project --no-hashes --no-dev \
      | grep -vE '^(torch|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==' > /tmp/requirements.txt \
    && uv pip install --system --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && uv pip install --system -r /tmp/requirements.txt

COPY src ./src
RUN uv pip install --system --no-deps .

# Non-root, as a PodSecurity "restricted" profile expects.
RUN useradd --uid 10001 --create-home app
USER 10001
EXPOSE 8080
CMD ["uvicorn", "--factory", "laya_router.service:create_app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
