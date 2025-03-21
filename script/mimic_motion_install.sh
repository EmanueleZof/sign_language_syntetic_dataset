#!/bin/bash
# Install MimicMotion
# https://github.com/Tencent/MimicMotion
################################################################################

echo "Test1"

#---- Install dependencies ----#
_AV_12="12.0.0"
_AV_13="13.1.0"

pip install decord onnxruntime omegaconf "av==$_AV_12"