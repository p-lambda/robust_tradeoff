#!/bin/bash
set -x

CODALAB_WKSHT=YOUR_WKSHT_HERE
mem=48g
gpus=2
cpus=2

epsilon=0.031
model=wrn-28-10

######################
# CIFAR STANDARD, AT, RST
#########################

beta=0.5

for trial in 1 2
do
take_amount_seed=157131${trial}7
seed=1893${trial}1

for take_fraction in 1.0
do
clean_epochs=$(echo "200 * $take_fraction" | bc -l)
clean_epochs=${clean_epochs%.*}

echo $clean_epochs

cl run -n table1-pgat-trial${trial} -w $CODALAB_WKSHT --request-docker-image sangxie513/robust-tradeoff-codalab-gpu --request-gpus ${gpus} --request-memory ${mem} --request-cpus ${cpus} data:data code:code train_pipeline_pgat.sh:train_pipeline_pgat.sh train_cleanmodel.sh:train_cleanmodel.sh "bash train_cleanmodel.sh ${take_fraction} ${take_amount_seed} ${clean_epochs} 0.0 0.0 ${seed} ${model} && bash train_pipeline_pgat.sh ${take_fraction} ${clean_epochs} ${epsilon} 0.0 ${beta} ${take_amount_seed} ${seed} 0.0 ${model} && bash train_pipeline_pgat.sh ${take_fraction} ${clean_epochs} ${epsilon} 0.9 ${beta} ${take_amount_seed} ${seed} 0.5 ${model}"
done
done

