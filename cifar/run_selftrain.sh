#!/bin/bash
epsilon=0.031
model=wrn-28-10

######################
# SELF TRAIN
#########################
beta=0.0
unlabeled_robust_weight=0.0

trial=1
take_fraction=1.0
take_amount_seed=157131${trial}7
seed=1893${trial}1

clean_epochs=$(echo "200 * $take_fraction" | bc -l)
clean_epochs=${clean_epochs%.*}

echo $clean_epochs

unlabeled_natural_weight=0.0
bash train_cleanmodel.sh ${take_fraction} ${take_amount_seed} ${clean_epochs} ${unlabeled_natural_weight} ${unlabeled_robust_weight} ${seed} ${model}


unlabeled_natural_weight=0.9
bash train_pipeline_pgat.sh ${take_fraction} ${clean_epochs} ${epsilon} ${unlabeled_natural_weight} ${beta} ${take_amount_seed} ${seed} ${unlabeled_robust_weight} ${model}
