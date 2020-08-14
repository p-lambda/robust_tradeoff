#!/bin/bash
set -x

CODALAB_WKSHT=YOUR_WKSHT_HERE
mem=32g
cpus=2
beta=0.5
model=wrn-40-2

for take_fraction in 0.1 0.125 0.2 0.5 1.0
do
    clean_epochs=$(echo "200 * $take_fraction" | bc -l)
    clean_epochs=${clean_epochs%.*}

    for trial in 1 2
    do
        clean_bundle_name=sample-sizes-clean-labeledproportion${take_fraction}-trial${trial}
        for epsilon in 0.0039 0.0078 0.0118 0.0157
        do
            cl run -m -n sample-sizes-eps${epsilon}-labeledproportion${take_fraction}-trial${trial} -w $CODALAB_WKSHT --request-queue tag=nlp --request-docker-image sangxie513/robust-tradeoff-codalab-gpu --request-gpus 1 --request-memory ${mem} --request-cpus ${cpus} data:data code:code train_pipeline.sh:train_pipeline.sh run_for_sample_size.sh:run_for_sample_size.sh clean_res:${clean_bundle_name} --- bash run_for_sample_size.sh ${epsilon} ${take_fraction} ${trial}
        done
    done
done
