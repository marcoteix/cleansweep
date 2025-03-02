#!/bin/bash

set -e

VERSION=0.1.0

echo Building the Docker image...
docker build -f docker/Dockerfile . -t marcoteix/cleansweep:$VERSION 

echo Pushing to Docker Hub...
docker push marcoteix/cleansweep:$VERSION

echo Done!
