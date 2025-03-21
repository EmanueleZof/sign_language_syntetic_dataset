#!/bin/bash
# Install MimicMotion
# https://github.com/Tencent/MimicMotion
################################################################################

if (( $# < 1 ))
then
	echo "USAGE: $0 <argument1> <argument2> ..."
	exit
fi

echo "Test1"

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