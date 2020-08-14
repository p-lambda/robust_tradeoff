#!/bin/bash
mem=16g
cpus=2

for take_fraction in 0.1 0.125 0.2 0.5 1.0
do
    for trial in 1 2
    do
        cl run -m -n sample-sizes-clean-labeledproportion${take_fraction}-trial${trial} -w robustness-tradeoff-paper-temp --request-queue tag=nlp --request-docker-image sangxie513/robust-tradeoff-codalab-gpu --request-gpus 1 --request-memory ${mem} --request-cpus ${cpus} data:data code:code run_clean_for_sample_size.sh:run_clean_for_sample_size.sh train_cleanmodel.sh:train_cleanmodel.sh --- bash run_clean_for_sample_size.sh ${take_fraction} ${trial}
    done
done

