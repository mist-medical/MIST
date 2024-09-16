ARG FROM_IMAGE_NAME=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
FROM ${FROM_IMAGE_NAME}

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install MIST.
RUN pip install --upgrade pip \
    && pip install --upgrade --no-cache-dir mist-medical

# Create app directory.
RUN mkdir /app
WORKDIR /app