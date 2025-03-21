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
}

for ARG in "$@"
do
	shift
	case "${ARG}" in
        "av13") 
            dependencies
            av13
            ;;
		*) 
            dependencies
            av12
            ;;
	esac
done