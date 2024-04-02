FROM python:3.10.13-slim-bookworm

RUN apt-get update \
    && apt-get --no-install-recommends install -y cmake build-essential \
    && apt-get clean

RUN pip install --upgrade pip
RUN pip install pytorch-lightning==2.1.0 torchmetrics==1.2.0
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install mlflow==2.7.2

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
