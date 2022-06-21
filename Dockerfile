FROM python:3.9-slim-bullseye
ENV PYTHONUNBUFFERED 1

ENV SHELL /bin/bash

ARG REQUIREMENTS

WORKDIR /project

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y wget tar unzip && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir model

WORKDIR /project/model

RUN wget -O model.zip https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip && unzip model.zip && rm model.zip

WORKDIR /project

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .


WORKDIR /project

EXPOSE ${PORT:-8000}/tcp

CMD gunicorn deepbooru:app \
    --name web \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers ${GUNICORN_WORKERS:-1} \
    --threads ${GUNICORN_THREADS:-4}  \
    --timeout ${GUNICORN_TIMEOUT:-60} \
    --capture-output \
    --enable-stdio-inheritance \
    --max-requests 1000  \
    --max-requests-jitter 100 \
    --keep-alive ${GUNICORN_TIMEOUT:-60} \
    ${GUNICORN_EXTRA_ARGS} \
    --log-level debug
