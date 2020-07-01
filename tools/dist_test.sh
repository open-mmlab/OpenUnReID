#!/usr/bin/env bash

set -x

PYTHON=${PYTHON:-"python"}

RESUME=$1
CONFIG=$2
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
test_reid.py ${RESUME} --config=${CONFIG} \
    --launcher="pytorch" --tcp-port=${PORT} --set ${PY_ARGS}
