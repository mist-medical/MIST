ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install MIST.
RUN pip install --upgrade pip \
    && pip install --no-cache-dir "mist-medical[train]"

# Create app directory.
RUN mkdir /app
WORKDIR /app