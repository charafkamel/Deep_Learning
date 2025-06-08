runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:v0.1 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home test-job --command -- echo "Hello"

runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:v0.2 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home test-job --command -- sh -c "cd ../../pvc/home/Deep_Learning/Project/Deep_Learning/ && sleep 20000"

runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:v0.2 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home test-job --command -- sh -c "cd ../../pvc/home/Deep_Learning/Project/Deep_Learning/ && pip install flash-attn==2.7.3 && pip install sentencepiece && python sft_training.py"

runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home test-job --command -- sh -c "cd ../../pvc/home/Deep_Learning/Project/Deep_Learning/ && pip install flash-attn==2.7.3 && pip install sentencepiece && sleep 72000"



### Run training

## base
runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest --gpu 1 --backoff-limit 0 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home base --command -- sh -c "pip install sentencepiece && cd ../../pvc/home/Deep_Learning/ && python main.py --base"

## count
runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest --gpu 1 --backoff-limit 0 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home count --command -- sh -c "pip install sentencepiece && cd ../../pvc/home/Deep_Learning/ && python main.py --count"


## base_generative
runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest --gpu 1 --backoff-limit 0 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home base-generative --command -- sh -c "pip install sentencepiece && cd ../../pvc/home/Deep_Learning/ && python main.py --base_generative"

## count_generative
runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest --gpu 1 --backoff-limit 0 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home count-generative --command -- sh -c "pip install sentencepiece && cd ../../pvc/home/Deep_Learning/ && python main.py --count_generative"

## rl
runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest --gpu 1 --backoff-limit 0 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home rl --command -- sh -c "pip install sentencepiece && cd ../../pvc/home/Deep_Learning/ && python main.py --rl"


### Sleeping job
runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest --gpu 1 --backoff-limit 0 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home debug --command -- sh -c "pip install sentencepiece && cd ../../pvc/home/Deep_Learning/ && sleep 72000"


### Run evaluation
runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest --gpu 1 --backoff-limit 0 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home evaluation --command -- sh -c "pip install sentencepiece && cd ../../pvc/home/Deep_Learning/ && python eval.py"