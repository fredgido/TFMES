FROM python:3.9-slim-bullseye
ENV PYTHONUNBUFFERED 1

ENV SHELL /bin/bash

ARG REQUIREMENTS

WORKDIR /project

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y wget tar unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir model

WORKDIR /project/model

RUN wget -O model.zip https://github.com/KichangKim/DeepDanbooru/releases/download/v4-20200814-sgd-e30/deepdanbooru-v4-20200814-sgd-e30.zip && unzip model.zip && rm model.zip

WORKDIR /project

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .


WORKDIR /project

CMD gunicorn deepbooru:app \
    --name web \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers ${GUNICORN_WORKERS:-2} \
    --threads ${GUNICORN_THREADS:-2}  \
    --timeout ${GUNICORN_TIMEOUT:-60} \
    --capture-output \
    --enable-stdio-inheritance \
    --max-requests 500  \
    --max-requests-jitter 100 \
    --keep-alive ${GUNICORN_TIMEOUT:-60} \
    --preload \
    ${GUNICORN_EXTRA_ARGS} \
    --log-level debug
