FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt update && apt install g++ -y

RUN pip install einops tiktoken