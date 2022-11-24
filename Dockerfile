FROM --platform=linux/amd64 python:3.8.12-slim

COPY api /api
COPY requirements_docker.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT