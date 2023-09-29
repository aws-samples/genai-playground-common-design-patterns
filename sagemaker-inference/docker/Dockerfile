FROM public.ecr.aws/ubuntu/ubuntu:20.04

RUN apt-get update -y && apt-get install -y python3.9 python3.9-distutils curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py
RUN pip3 install "fschat[webui,model_worker]"
RUN pip3 install boto3 jsonpath-ng
ENV HUGGINGFACE_HUB_CACHE=/tmp
COPY serve /root/genai-playground-common-design-patterns/serve
