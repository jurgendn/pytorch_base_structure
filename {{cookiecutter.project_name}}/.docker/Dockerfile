FROM python:3.10.10-slim-bullseye

RUN apt-get update

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd -m working
USER working
WORKDIR /home/working

COPY . .

RUN ['bash', ".docker/startup.sh"]