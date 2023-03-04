#!/bin/bash

maxThreads=$1
useLock=$2
sizeStart=$3
sizeFinish=$4
sizeStep=$5

if [[ ${useLock} -eq 0 ]]
then
    lock=""
elif [[ ${useLock} -eq 1 ]]
then
    lock="lock"
fi

scheduals=("static" "dynamic" "guided")

for ((size=${sizeStart}; size<=${sizeFinish}; size+=${sizeStep}))
do
    flags="-size ${size} -threads ${maxThreads}"
    run="./run ${flags}"
    for schedual in "${scheduals[@]}"
    do
        maker="${schedual}${lock}"
        make ${maker}
        echo $maker
        ${run}
        make clean
    done
done

