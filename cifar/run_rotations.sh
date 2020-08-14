#!/bin/bash
epsilon=0.031

#############
# ROTATIONS
#############
# unlabeled_robust_weight=0.0
beta=0.5
model=wrn-40-2

for trial in 1 2
do
take_amount_seed=157131${trial}7
seed=1893${trial}1

take_fraction=1.0
clean_epochs=$(echo "200 * $take_fraction" | bc -l)
clean_epochs=${clean_epochs%.*}

echo $clean_epochs

# standard training
unlabeled_natural_weight=0.0
unlabeled_robust_weight=0.0
bash train_cleanmodel.sh ${take_fraction} ${take_amount_seed} ${clean_epochs} ${unlabeled_natural_weight} ${unlabeled_robust_weight} ${seed} ${model}

# Adversarial training
unlabeled_natural_weight=0.0
unlabeled_robust_weight=0.0
# worst of 10
bash train_pipeline_rotations.sh ${take_fraction} ${clean_epochs} ${unlabeled_natural_weight} ${beta} ${take_amount_seed} ${seed} ${unlabeled_robust_weight} ${model} spatial
# random
bash train_pipeline_rotations.sh ${take_fraction} ${clean_epochs} ${unlabeled_natural_weight} ${beta} ${take_amount_seed} ${seed} ${unlabeled_robust_weight} ${model} spatial_random

# RST
unlabeled_natural_weight=0.9
unlabeled_robust_weight=0.5
bash train_pipeline_rotations.sh ${take_fraction} ${clean_epochs} ${unlabeled_natural_weight} ${beta} ${take_amount_seed} ${seed} ${unlabeled_robust_weight} ${model} spatial
bash train_pipeline_rotations.sh ${take_fraction} ${clean_epochs} ${unlabeled_natural_weight} ${beta} ${take_amount_seed} ${seed} ${unlabeled_robust_weight} ${model} spatial_random
done

