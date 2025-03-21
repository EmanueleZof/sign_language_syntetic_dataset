#!/bin/bash
# Install MimicMotion
# https://github.com/Tencent/MimicMotion
################################################################################

echo "***** Installing MimicMotion *****"

#---- libraries ----#
pip install decord onnxruntime omegaconf

if [ $1="av13" ]
then
    pip install av==13.1.0
else
    pip install av==12.0.0
fi