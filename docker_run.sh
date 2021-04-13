#!/bin/bash

set -e
set -x

docker run \
-p 5000:5000 \
-it --rm --name speech-2-text speech-2-text
