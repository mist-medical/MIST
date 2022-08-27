ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.05-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN pip install nvidia-pyindex
RUN pip install --upgrade pip
RUN pip install tensorflow-addons --upgrade
RUN pip install antspyx --upgrade
RUN pip install SimpleITK --upgrade

ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.3
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV OMPI_MCA_coll_hcoll_enable 0
ENV HCOLL_ENABLE_MCAST 0