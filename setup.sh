#!/bin/bash

if [ `echo $PATH | grep -o keras` == "keras" ]; then
    echo "keras running, good";
else
    echo "keras not running, quitting";
    return
fi

echo 'Fetching external packages'

git submodule update --init


echo 'appending python path'

export PYTHONPATH=$PYTHONPATH:$PWD/DLKit


