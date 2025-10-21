# ===== Base: matches IceCube image, Python already present =====
ARG BASE_IMAGE=icecube/icetray:icetray-devel-v1.13.0-ubuntu22.04-2025-02-12
FROM ${BASE_IMAGE}

# ===== Homogenized build args (same names as your other Dockerfile) =====
# NOTE: Python is already in the base image; PYTHON_VERSION is accepted but unused.
ARG TORCH_CHOICE="torch-2.6.0"   # allowed: torch-2.7.0 | torch-2.6.0 | torch-2.5.1 | no_torch
ARG HARDWARE="cpu"               # forced CPU-only in this image
ARG PYTHON_VERSION="3.11"        # accepted, not used (kept for compatibility)
ARG GRAPHNET_REF="main"

# Keep logs quiet/predictable
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONNOUSERSITE=1 \
    PIP_NO_CACHE_DIR=1

# ===== Match CI/Icetray environment =====
ENV I3_SRC=/opt/icetray/share/icetray \
    I3_BUILD=/opt/icetray/share/icetray \
    I3_TESTDATA=/root/icetray/build/test-data \
    I3_PRODDATA=/root/icetray/build/prod-data \
    ROOTSYS=/opt/icetray/cernroot \
    HDF5_USE_FILE_LOCKING=FALSE

ENV LD_LIBRARY_PATH=/opt/icetray/lib:/opt/icetray/cernroot/lib:/opt/icetray/lib/tools:${LD_LIBRARY_PATH}
ENV DYLD_LIBRARY_PATH=/opt/icetray/lib:/opt/icetray/cernroot/lib:/opt/icetray/lib/tools:${DYLD_LIBRARY_PATH}
ENV PYTHONPATH=/opt/icetray/lib:/opt/icetray/cernroot/lib:${PYTHONPATH}
ENV PATH=/opt/icetray/bin:${PATH}


# ===== OS deps (minimal) =====
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates python3-venv build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*

# ===== Create & activate venv (keep your pattern; do NOT install Python itself) =====
WORKDIR /opt/app
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

# Ensure pip/setuptools are recent (matches “mature format”)
RUN python -m pip install --no-cache-dir --upgrade "pip>=23" "setuptools>=68.2.2" "wheel"

# -------- Install PyTorch (CPU-only, mature logic & guards) --------
RUN set -eux; \
  echo "TORCH_CHOICE=${TORCH_CHOICE} HARDWARE=${HARDWARE}"; \
  # Fail fast if non-CPU requested for this image family
  case "$HARDWARE" in \
    cpu) ;; \
    *) echo "This Dockerfile only supports CPU. HARDWARE=$HARDWARE is not allowed." >&2; exit 11 ;; \
  esac; \
  if [ "$TORCH_CHOICE" = "no_torch" ]; then \
    echo "Skipping PyTorch install"; \
  else \
    case "$TORCH_CHOICE" in \
      torch-2.7.0) TV=2.7.0 ;; \
      torch-2.6.0) TV=2.6.0 ;; \
      torch-2.5.1) TV=2.5.1 ;; \
      *) echo "Unsupported TORCH_CHOICE=$TORCH_CHOICE" >&2; exit 1 ;; \
    esac; \
    # In your GPU-friendly Dockerfile you map HARDWARE→index URL; here we force CPU:
    IDX=https://download.pytorch.org/whl/cpu; \
    python -m pip install --no-cache-dir "torch==${TV}" torchvision torchaudio --index-url "${IDX}"; \
  fi

# -------- Clone GraphNeT at requested ref (robust ref resolution) --------
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

# -------- Install GraphNeT + extras (matches mature mapping, CPU-only wheel URL) --------
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
    WHEEL_URL="https://data.pyg.org/whl/torch-${TV}+cpu.html"; \
    python -m pip install --no-cache-dir -e ".[${EX}]" -f "${WHEEL_URL}"; \
  fi

# Optional: jammy_flows (kept to match mature format)
RUN python -m pip install --no-cache-dir "git+https://github.com/thoglu/jammy_flows.git"

# -------- Labels & entrypoint --------
LABEL org.opencontainers.image.source="https://github.com/graphnet-team/graphnet" \
      org.opencontainers.image.version="${GRAPHNET_REF}"

# ===== Runtime defaults =====
ENTRYPOINT ["/bin/bash", "-lc", "python -c 'import graphnet, torch, torch_geometric,sys; print(\"GraphNeT:\", graphnet.__version__);print(\"torch:\", torch.__version__); print(\"torch_geometric:\", torch_geometric.__version__)'; python3 --version;  exec bash"]
