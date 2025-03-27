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
cd sign_language_syntetic_dataset/
git clone https://github.com/Tencent/MimicMotion.git

#---- Create folders ----#
#cd MimicMotion/
mkdir models
mkdir -p models/DWPose

#---- Load models ----#
wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx

wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx

wget https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth -P models/

