# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---- System deps for building your C++/CMake parts ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libbz2-dev \
    libz-dev \
    libicu-dev \
    libboost-all-dev

# ---- Copy your repo (no git clone needed) ----
WORKDIR /jani_env
COPY . /jani_env

# ---- Python deps ----
# (PyTorch image already has python + torch; you install your extras.)
# RUN python -m venv /opt/venv
# ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir --break-system-packages -r requirements_training.txt

# ---- Build the C++ engine ----
WORKDIR /jani_env/jani/engine
RUN mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j"$(nproc)"

# ---- Runtime env ----
WORKDIR /jani_env
ENV PYTHONPATH=/jani_env

# Optional quick self-check (comment out after debugging):
RUN python -c "import mask_ppo.train; import dagger.train; print('Import OK')"