FROM python:3.11-slim-bookworm

ENV GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no"

RUN apt-get update \
    && apt-get install -y libgl1 libglib2.0-0 curl wget git procps \
    && rm -rf /var/lib/apt/lists/*

# Install core dependencies
RUN pip install --no-cache-dir torch transformers sentence-transformers --extra-index-url https://download.pytorch.org/whl/cpu

ENV HF_HOME=/tmp/
ENV TORCH_HOME=/tmp/

COPY requirements.txt /root/requirements.txt
RUN pip install --no-cache-dir -r /root/requirements.txt

# On container environments, always set a thread budget to avoid undesired thread congestion.
ENV OMP_NUM_THREADS=4

# Copy application files
COPY . /root/app/
WORKDIR /root/app/
