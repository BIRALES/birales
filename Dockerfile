FROM ubuntu:22.04

RUN apt-get update \
  && apt-get install --assume-yes --no-install-recommends --quiet \
        python3  python3-pip \
  && apt-get clean all
RUN pip install --no-cache --upgrade pip setuptools

RUN apt-get install --assume-yes redis git
RUN pip install git+https://gitlab.com/ska-telescope/pyfabil.git

COPY python /app

WORKDIR app
RUN pip install .
RUN pip install -r requirements_birales.txt