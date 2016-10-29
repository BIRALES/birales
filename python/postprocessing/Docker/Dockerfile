# Dockerfile to build Birales-Post

# Set the base image to Ubuntu
FROM ubuntu:14.04

# File Author / Maintainer
MAINTAINER Denis Cutajar

# Add the application resources URL
RUN echo "deb http://archive.ubuntu.com/ubuntu/ $(lsb_release -sc) main universe" >> /etc/apt/sources.list

RUN apt-get update

# Install basic applications
RUN apt-get install -y tar git curl nano wget

# Install Python and Basic Python Tools
RUN apt-get install -y python python-dev python-distribute python-pip

# Install Mongo DB
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10
RUN echo "deb http://repo.mongodb.org/apt/ubuntu "$(lsb_release -sc)"/mongodb-org/3.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-3.0.list
RUN apt-get install -y mongodb-org

# Update the sources list
RUN apt-get update

# Application Deployment
RUN mkdir /data
RUN mkdir /birales
VOLUME /data
WORKDIR /birales
RUN git clone https://deniscutajar@bitbucket.org/lessju/birales.git

# Get pip to download and install requirements:
RUN pip install -r /birales/python/postprocessing/requirements.txt

# Expose ports
EXPOSE 80
