#!/usr/bin/env bash

set -x

PYTHON=${PYTHON:-"python"}

METHOD=$1
WORK_DIR=$2
PY_ARGS=${@:3}

GPUS=${GPUS:-4}

while true # find unused tcp port
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
$METHOD/main.py $METHOD/config.yaml --work-dir=${WORK_DIR} \
    --launcher="pytorch" --tcp-port=${PORT} --set ${PY_ARGS}
