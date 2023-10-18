FROM python:3.10.13-slim-bookworm

RUN apt-get update
RUN apt-get install -y cmake build-essential

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
