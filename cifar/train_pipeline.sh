#!/bin/bash
set -x
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PATH="$PATH:/usr/local/cuda-10.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64"

take_fraction=$1
clean_epochs=$2 # 1 epoch is number of steps wrt 50K samples
epsilon=$3
unlabeled_natural_weight=$4
beta=$5
take_amount_seed=$6
seed=$7
unlabeled_robust_weight=$8
model=$9

clean_beta=0
ds=1
batch_size=128
lr=cosine
unsup_frac=0.5

clean_name="${model}-take_frac=${take_fraction}-clean-take_seed=${take_amount_seed}-epochs=${clean_epochs}"

clean_output_dir="clean_res/clean_models/${clean_name}"
data_dir="data"
unlabeled_targets_dir="clean_res/unlabeled_targets"
aux_targets_filename="${unlabeled_targets_dir}/targets-model=${model}-take_frac=${take_fraction}-take_seed=${take_amount_seed}.pickle"

# Final adversarial training 
epochs=200
ds=1
unsup_frac=0.5
batch_size=256
topk=50
kw=_v3.1

weight_decay=5e-4
lr=cosine

flags=--add_aux_labels
name="${model}-take_frac=${take_fraction}-take_seed=${take_amount_seed}-unw=${unlabeled_natural_weight}-urw=${unlabeled_robust_weight}-seed=${seed}-beta=${beta}-epsilon=${epsilon}"
echo Training ${name}

mkdir -p models
output_dir="models/${name}"

num_steps=10

python $DIR/code/train_trades_cifar10_semisup.py \
    --model ${model} \
    --epochs ${epochs} --batch_size ${batch_size} \
    --save_freq 200 \
    --train_downsample ${ds} \
    --log_interval 20 \
    --model_dir ${output_dir} \
    --seed ${seed} \
    --beta ${beta} \
    --distance l_inf \
    --test_batch_size 500 \
    --eval_attack_batches 1 \
    --unsup_fraction ${unsup_frac} \
    --data_dir ${data_dir} \
    --aux_data_filename ti_top_$((topk * 1000))_pred${kw}.pickle \
    --aux_targets_filename ${aux_targets_filename}\
    ${flags} --lr_schedule ${lr} --eval_freq 1 --nesterov \
    --weight_decay ${weight_decay} \
    --unlabeled_robust_weight ${unlabeled_robust_weight} \
    --unlabeled_natural_weight ${unlabeled_natural_weight} \
    --loss 'madry' \
    --train_take_fraction ${take_fraction} \
    --take_amount_seed ${take_amount_seed} \
    --epsilon ${epsilon} \
    --num-steps ${num_steps} \

# Running final attack 

echo Test with final attack

num_steps=40
step_size=0.01
num_restarts=5

checkpoint_path="${output_dir}/checkpoint-epoch${epochs}.pt"
attack=pgd
output_suffix="eps=${epsilon}-steps=${num_steps}-ss=${step_size}-restarts=${num_restarts}"
python $DIR/code/attack_cifar10.py \
       --model_path ${checkpoint_path} \
       --model ${model} \
       --num_steps ${num_steps} \
       --step_size ${step_size} \
       --num_restarts ${num_restarts} \
       --epsilon ${epsilon} \
       --output_suffix ${output_suffix} \
       --attack ${attack}
