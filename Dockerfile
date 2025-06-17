FROM pytorch/pytorch:2.7.1-cuda12.4-cudnn9-runtime

# Set environment variables for non-interactive installation.
ENV DEBIAN_FRONTEND=noninteractive

# Install MIST.
RUN pip install --upgrade pip \
    && pip install --upgrade --no-cache-dir mist-medical

# Create app directory.
RUN mkdir /app
WORKDIR /app