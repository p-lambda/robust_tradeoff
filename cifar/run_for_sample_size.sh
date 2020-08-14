#!/bin/bash
set -x

epsilon=$1
take_fraction=$2
trial=$3

beta=0.5
model=wrn-40-2

take_amount_seed=157131${trial}7
seed=1893${trial}1
clean_epochs=$(echo "200 * $take_fraction" | bc -l)
clean_epochs=${clean_epochs%.*}

# Adversarial training
unlabeled_natural_weight=0.0
unlabeled_robust_weight=0.0
bash train_pipeline.sh ${take_fraction} ${clean_epochs} ${epsilon} ${unlabeled_natural_weight} ${beta} ${take_amount_seed} ${seed} ${unlabeled_robust_weight} ${model} "clean_res/"

# RST
if (( $(echo "$take_fraction == 1.0" |bc -l) )); then
    unlabeled_natural_weight=0.9
else
    unlabeled_natural_weight=${take_fraction}
fi
unlabeled_robust_weight=0.5
bash train_pipeline.sh ${take_fraction} ${clean_epochs} ${epsilon} ${unlabeled_natural_weight} ${beta} ${take_amount_seed} ${seed} ${unlabeled_robust_weight} ${model} "clean_res/"

