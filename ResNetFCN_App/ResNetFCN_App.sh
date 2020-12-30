#!/bin/bash


echo "$1"

docker run -e IMG_PATH="$1" -v $(pwd)/output:/usr/src/app/output -v $(pwd)/"$1":/usr/src/app/"$1" resnetfcn_app
