ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.06-py3
FROM ${FROM_IMAGE_NAME}

# Set environment variables 
#ENV OMP_NUM_THREADS=2
#ENV OMPI_MCA_coll_hcoll_enable 0
#ENV HCOLL_ENABLE_MCAST 0
ENV DEBIAN_FRONTEND noninteractive

# Install mist
RUN pip install --upgrade pip \
    && pip install --upgrade --no-cache-dir mist-medical

# Install dependencies
#RUN apt-get update -y --fix-missing \
#    && apt-get install -y cmake git
#
## Install ANTs
#RUN mkdir /opt/ants \
#    && cd /opt/ants \
#    && git clone https://github.com/ANTsX/ANTs.git \
#    && cd /opt/ants/ANTs \
#    && git checkout v2.5.0 \
#    && cd /opt/ants \
#    && mkdir build install \
#    && cd /opt/ants/build \
#    && cmake -DCMAKE_INSTALL_PREFIX=/opt/ants/install ../ANTs 2>&1 | tee cmake.log \
#    && make -j 4 2>&1 | tee build.log \
#    && cd /opt/ants/build/ANTS-build \
#    && make install 2>&1 | tee install.log \
#
## Install c3d
#RUN mkdir /opt/c3d \
#    && cd /opt/c3d/ \
#    && wget https://downloads.sourceforge.net/project/c3d/c3d/Nightly/c3d-nightly-Linux-x86_64.tar.gz \
#    && tar -xvf c3d-nightly-Linux-x86_64.tar.gz \
#    && cp c3d-1.1.0-Linux-x86_64/bin/c3d /usr/local/bin/ \

# Create working directory
#ENV PATH /opt/ants/install/bin:$PATH
RUN mkdir /app
WORKDIR /app