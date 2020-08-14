#!/bin/bash
set -x

take_fraction=$1
trial=$2

beta=0.5
model=wrn-40-2

take_amount_seed=157131${trial}7
seed=1893${trial}1
clean_epochs=$(echo "200 * $take_fraction" | bc -l)
clean_epochs=${clean_epochs%.*}

# Standard training
unlabeled_natural_weight=0.0
unlabeled_robust_weight=0.0
bash train_cleanmodel.sh ${take_fraction} ${take_amount_seed} ${clean_epochs} ${unlabeled_natural_weight} ${unlabeled_robust_weight} ${seed} ${model}
echo $jid
