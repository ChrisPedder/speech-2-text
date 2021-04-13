#!/bin/bash

set -x
set -e

docker build . -t speech-2-text $@
