#!/bin/bash
# Install MimicMotion on Google Colab
# https://github.com/Tencent/MimicMotion
################################################################################

echo "***** Installing MimicMotion *****"

#---- Install libraries ----#
pip install decord onnxruntime omegaconf

if [[ $1 == av13 ]]
then
    pip install av==13.1.0
else
    pip install av==12.0.0
fi

#---- Clone Git repository ----#
git clone https://github.com/Tencent/MimicMotion.git