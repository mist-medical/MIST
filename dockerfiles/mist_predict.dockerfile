ARG FROM_IMAGE_NAME=mistmedical/mist:latest
FROM ${FROM_IMAGE_NAME}

ENTRYPOINT ["mist_predict"]