FROM python:3.12
# FROM kaggle/python
MAINTAINER Shun

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install transformers
RUN pip install torch
RUN pip install matplotlib
RUN pip install accelerate>=0.26.0
RUN pip install jupyterlab==3.4.6

RUN pip install ipywidgets --upgrade
RUN pip install jupyter --upgrade
RUN jupyter nbextension enable --py widgetsnbextension --sys-prefix
RUN jupyter nbextension install --py widgetsnbextension --sys-prefix

RUN jupyter serverextension enable --py jupyterlab