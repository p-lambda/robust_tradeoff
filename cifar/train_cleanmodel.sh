#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PATH="$PATH:/usr/local/cuda-10.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64"

set -x

take_fraction=$1
take_amount_seed=$2
clean_epochs=$3 # 1 epoch is number of steps wrt 50K samples
unlabeled_natural_weight=$4
unlabeled_robust_weight=$5
seed=$6
model=$7

beta=0
epsilon=0.25
ds=1
batch_size=128
lr=cosine
unsup_frac=0.5

clean_name="${model}-take_frac=${take_fraction}-clean-take_seed=${take_amount_seed}-epochs=${clean_epochs}"

echo Training ${clean_name}

mkdir -p clean_models
mkdir -p unlabeled_targets
clean_output_dir="clean_models/${clean_name}"
data_dir="data"
unlabeled_targets_dir="unlabeled_targets"
aux_targets_filename="${unlabeled_targets_dir}/targets-model=${model}-take_frac=${take_fraction}-take_seed=${take_amount_seed}.pickle"

if [ ! -f "${aux_targets_filename}" ];
then
    python $DIR/code/train_trades_cifar10_semisup.py \
        --model ${model} \
        --epochs ${clean_epochs} --batch_size ${batch_size} \
        --save_freq 200 \
        --log_interval 100 \
        --model_dir ${clean_output_dir} \
        --beta ${beta} \
        --epsilon ${epsilon} \
        --distance l_inf \
        --num-steps 0 \
        --test_batch_size 500 \
        --eval_attack_batches 0 \
        --unsup_fraction ${unsup_frac} \
        --data_dir ${data_dir} \
        --lr_schedule ${lr} --eval_freq 1 --nesterov \
        --weight_decay 5e-4 --normalize_input \
        --train_take_fraction ${take_fraction} \
        --seed ${seed} \
        --take_amount_seed ${take_amount_seed} \

    echo Generate labels

    # Generate labels
    python $DIR/code/generate_labels.py  \
        --model_dir ${clean_output_dir} \
        --data_dir ${data_dir} \
        --output_dir ${unlabeled_targets_dir} \
        --model ${model} \
        --model_epoch ${clean_epochs} \
        --take_fraction ${take_fraction} \
        --take_amount_seed ${take_amount_seed}
fi
