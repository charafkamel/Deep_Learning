#!/bin/bash

# Common settings
IMAGE="registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest"
GPU=1
PVC1="course-ee-559-scratch:/scratch"
PVC2="home:/pvc/home"
WORKDIR="../../pvc/home/Deep_Learning"
PIP_INSTALL="pip install sentencepiece"

submit_job() {
    local NAME=$1
    local CMD=$2

    echo "Submitting job: $NAME"
    runai submit \
        --image "$IMAGE" \
        --gpu "$GPU" \
        --backoff-limit 0 \
        --pvc "$PVC1" \
        --pvc "$PVC2" \
        "$NAME" \
        --command -- sh -c "$PIP_INSTALL && cd $WORKDIR && $CMD"
}

# Submit each job
submit_job base "python main.py --base"
submit_job count "python main.py --count"
submit_job base-generative "python main.py --base_generative"
submit_job count-generative "python main.py --count_generative"
submit_job rl "python main.py --rl"
#submit_job evaluation "python eval/eval.py"
