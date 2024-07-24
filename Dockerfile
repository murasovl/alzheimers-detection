# TODO: select a base image
# Tip: start with a full base image, and then see if you can optimize with
#      a slim or tensorflow base

#      Standard version

# FROM python:3.10.6


#      Slim version
# FROM python:3.10-slim

#      Tensorflow version
FROM tensorflow/tensorflow:2.15.0

WORKDIR /prod


#      Or tensorflow to run on Apple Silicon (M1 / M2)
# FROM armswdev/tensorflow-arm-neoverse:r23.08-tf-2.13.0-eigen

RUN apt-get update && apt-get install -y libgl1-mesa-glx

COPY requirements.txt requirements_docker.txt
RUN pip install -r requirements_docker.txt
# Copy everything we need into the image
COPY alzheimers_detection_tool alzheimers_detection_tool
COPY api api
#COPY scripts scripts

COPY setup.py setup.py
COPY models models
COPY shap shap
#COPY credentials.json credentials.json

# Install everything
RUN pip install --upgrade pip
RUN pip install .

# Make directories that we need, but that are not included in the COPY
#RUN mkdir /raw_data
#RUN mkdir /models

# TODO: to speed up, you can load your model from MLFlow or Google Cloud Storage at startup using
# RUN python -c 'replace_this_with_the_commands_you_need_to_run_to_load_the_model'

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
