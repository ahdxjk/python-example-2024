FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
RUN pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install PyWavelets
RUN pip install pytorch_wavelets
