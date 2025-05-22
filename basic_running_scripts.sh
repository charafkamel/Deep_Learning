runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:v0.1 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home test-job --command -- echo "Hello"

runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:v0.2 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home test-job --command -- sh -c "cd ../../pvc/home/DeepLearning/Project/Deep_Learning/ && sleep 20000"

runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:v0.2 --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home test-job --command -- sh -c "cd ../../pvc/home/DeepLearning/Project/Deep_Learning/ && pip install flash-attn==2.7.3 && pip install sentencepiece && python sft_training.py"

runai submit --image registry.rcp.epfl.ch/ee-559-charaf/my-toolbox:latest --gpu 1 --pvc course-ee-559-scratch:/scratch --pvc home:/pvc/home test-job --command -- sh -c "cd ../../pvc/home/DeepLearning/Project/Deep_Learning/ && pip install flash-attn==2.7.3 && pip install sentencepiece && sleep 72000"