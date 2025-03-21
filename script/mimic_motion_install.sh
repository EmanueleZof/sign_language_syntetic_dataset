#!/bin/bash
# Install MimicMotion
# https://github.com/Tencent/MimicMotion
################################################################################

echo "Installing MimicMotion"

#---- Dependencies ----#
av12()
{
    pip install av==12.0.0
}

av13()
{
    pip install av==13.1.0
}

dependencies()
{
    pip install decord onnxruntime omegaconf

    if [[ "$1" == "av13" ]]; then
        pip install av==13.1.0
    else
        pip install av==12.0.0
    fi
}

dependencies