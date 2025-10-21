# syntax=docker/dockerfile:1.7

# -------- Build args (pre-FROM) --------
ARG BASE_IMAGE=ubuntu:22.04

# -------- Base image --------
FROM ${BASE_IMAGE}

# -------- Re-declare build args (now in scope) --------
ARG TORCH_CHOICE=torch-2.6.0             # torch-2.7.0 | torch-2.6.0 | torch-2.5.1 | no_torch
ARG HARDWARE=cpu                          # cpu | cu118 | cu121 | cu124 | cu126 | cu128
ARG PYTHON_VERSION=3.11                   # used by Miniforge
ARG GRAPHNET_REF=main

# Make APT quiet & noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Expose args as envs
ENV TORCH_CHOICE=$TORCH_CHOICE \
    HARDWARE=$HARDWARE \
    PYTHON_VERSION=$PYTHON_VERSION \
    GRAPHNET_REF=$GRAPHNET_REF

# -------- System deps --------
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl git ca-certificates build-essential bash \
    && rm -rf /var/lib/apt/lists/*

# -------- Install Miniforge (for specific Python version) --------
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
 && rm /tmp/miniforge.sh \
 && $CONDA_DIR/bin/conda install -y python=${PYTHON_VERSION} \
 && $CONDA_DIR/bin/conda clean -afy

# Add conda to PATH
ENV PATH="$CONDA_DIR/bin:$PATH"

# Verify Python version
RUN python --version

# Ensure pip/setuptools are recent
RUN python -m pip install --no-cache-dir --upgrade "pip>=23" "setuptools>=68.2.2" "wheel"

# -------- Install PyTorch (if requested) --------
RUN set -eux; \
    echo "TORCH_CHOICE=$TORCH_CHOICE HARDWARE=$HARDWARE"; \
    if [ "$TORCH_CHOICE" = "no_torch" ]; then \
      echo "Skipping PyTorch install"; \
    else \
      case "$TORCH_CHOICE" in \
        torch-2.7.0) TV=2.7.0 ;; \
        torch-2.6.0) TV=2.6.0 ;; \
        torch-2.5.1) TV=2.5.1 ;; \
        *) echo "Unsupported TORCH_CHOICE=$TORCH_CHOICE" >&2; exit 1 ;; \
      esac; \
      case "$TORCH_CHOICE:$HARDWARE" in \
        torch-2.7.0:cu121|torch-2.7.0:cu124|\
        torch-2.6.0:cu121|torch-2.6.0:cu128|\
        torch-2.5.1:cu126|torch-2.5.1:cu128) \
          echo "Incompatible combo TORCH=$TORCH_CHOICE HARDWARE=$HARDWARE" >&2; exit 2 ;; \
      esac; \
      case "$HARDWARE" in \
        cpu)   IDX=https://download.pytorch.org/whl/cpu ;; \
        cu118) IDX=https://download.pytorch.org/whl/cu118 ;; \
        cu121) IDX=https://download.pytorch.org/whl/cu121 ;; \
        cu124) IDX=https://download.pytorch.org/whl/cu124 ;; \
        cu126) IDX=https://download.pytorch.org/whl/cu126 ;; \
        cu128) IDX=https://download.pytorch.org/whl/cu128 ;; \
        *) echo "Unsupported HARDWARE=$HARDWARE" >&2; exit 1 ;; \
      esac; \
      python -m pip install --no-cache-dir "torch==${TV}" torchvision torchaudio --index-url "${IDX}"; \
    fi

# -------- Clone GraphNeT at requested ref --------
WORKDIR /opt/graphnet
RUN set -eux; \
    REPO="https://github.com/graphnet-team/graphnet.git"; \
    git clone "$REPO" .; \
    git fetch --tags --quiet; \
    if [ "$GRAPHNET_REF" = "main" ]; then \
      git checkout -B buildref origin/main; \
    else \
      if git rev-parse -q --verify "refs/tags/v${GRAPHNET_REF}" >/dev/null; then \
        git checkout -B buildref "refs/tags/v${GRAPHNET_REF}"; \
      elif git rev-parse -q --verify "refs/tags/${GRAPHNET_REF}" >/dev/null; then \
        git checkout -B buildref "refs/tags/${GRAPHNET_REF}"; \
      elif git rev-parse -q --verify "origin/${GRAPHNET_REF}" >/dev/null; then \
        git checkout -B buildref "origin/${GRAPHNET_REF}"; \
      else \
        echo "GraphNeT ref '${GRAPHNET_REF}' not found (tried v${GRAPHNET_REF}, ${GRAPHNET_REF}, origin/${GRAPHNET_REF})" >&2; \
        exit 9; \
      fi; \
    fi

# -------- Install GraphNeT + extras --------
RUN set -eux; \
    case "$TORCH_CHOICE" in \
      torch-2.7.0) EX=torch-27 ;; \
      torch-2.6.0) EX=torch-26 ;; \
      torch-2.5.1) EX=torch-25 ;; \
      no_torch)    EX=develop  ;; \
      *) echo "Unsupported TORCH_CHOICE"; exit 1 ;; \
    esac; \
    if [ "$TORCH_CHOICE" = "no_torch" ]; then \
      python -m pip install --no-cache-dir -e ".[develop]"; \
    else \
      TV="${TORCH_CHOICE#torch-}"; \
      WHEEL_URL="https://data.pyg.org/whl/torch-${TV}+${HARDWARE}.html"; \
      python -m pip install --no-cache-dir -e ".[${EX}]" -f "${WHEEL_URL}"; \
    fi

# Optional: jammy_flows
RUN python -m pip install --no-cache-dir "git+https://github.com/thoglu/jammy_flows.git"

# -------- Labels & entrypoint --------
LABEL org.opencontainers.image.source="https://github.com/graphnet-team/graphnet" \
      org.opencontainers.image.version="${GRAPHNET_REF}"

ENTRYPOINT ["/bin/bash", "-lc", "python -c 'import graphnet, torch, torch_geometric,sys; print(\"GraphNeT:\", graphnet.__version__);print(\"torch:\", torch.__version__); print(\"torch_geometric:\", torch_geometric.__version__)'; python3 --version;  exec bash"]
