ARG FROM_IMAGE_NAME=pytorch/pytorch
FROM ${FROM_IMAGE_NAME}

RUN pip install nvidia-pyindex
RUN pip install --upgrade pip
RUN pip install antspyx
RUN pip install SimpleITK
RUN pip install monai
RUN pip install tensorboard
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# Use the following if running cuda 12
# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

ENV OMP_NUM_THREADS=2
ENV OMPI_MCA_coll_hcoll_enable 0
ENV HCOLL_ENABLE_MCAST 0

# setup dependencies 
RUN apt-get update

RUN apt-get install -y cmake git
# install ants
RUN mkdir /opt/ants
WORKDIR /opt/ants

RUN git clone https://github.com/ANTsX/ANTs.git
WORKDIR /opt/ants/ANTs
RUN git checkout v2.5.0
WORKDIR /opt/ants
RUN mkdir build install
WORKDIR /opt/ants/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/opt/ants/install ../ANTs 2>&1 | tee cmake.log
RUN make -j 4 2>&1 | tee build.log
WORKDIR /opt/ants/build/ANTS-build
RUN make install 2>&1 | tee install.log
# install c3d
RUN mkdir /opt/c3d
WORKDIR /opt/c3d/
RUN wget https://downloads.sourceforge.net/project/c3d/c3d/Nightly/c3d-nightly-Linux-x86_64.tar.gz
RUN tar -xvf c3d-nightly-Linux-x86_64.tar.gz
RUN cp c3d-1.1.0-Linux-x86_64/bin/c?d /usr/local/bin/

# env
ENV PATH /opt/ants/install/bin:$PATH
RUN apt-get install -y python2.7

RUN mkdir /mist

WORKDIR /mist
ADD . /mist
