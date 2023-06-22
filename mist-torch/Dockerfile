ARG FROM_IMAGE_NAME=pytorch/pytorch
FROM ${FROM_IMAGE_NAME}

RUN pip install nvidia-pyindex
RUN pip install --upgrade pip
RUN pip install antspyx
RUN pip install SimpleITK
RUN pip install monai
RUN pip install tenorboard
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

# Use the following if running cuda 12
# pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

ENV OMP_NUM_THREADS=2
ENV OMPI_MCA_coll_hcoll_enable 0
ENV HCOLL_ENABLE_MCAST 0

RUN mkdir /mist

WORKDIR /mist
ADD . /mist