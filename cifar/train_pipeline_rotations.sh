#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

set -x

export PATH="$PATH:/usr/local/cuda-10.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64"

take_fraction=$1
clean_epochs=$2 # 1 epoch is number of steps wrt 50K samples
unlabeled_natural_weight=$3
beta=$4
take_amount_seed=$5
seed=$6
unlabeled_robust_weight=$7
model=$8
distance=$9

max_rot=30
max_trans=0.1071

clean_beta=0
ds=1
batch_size=128
lr=cosine
unsup_frac=0.5

clean_name="${model}-take_frac=${take_fraction}-clean-take_seed=${take_amount_seed}-epochs=${clean_epochs}"

echo Training ${clean_name}

mkdir -p clean_models
mkdir -p unlabeled_targets
clean_output_dir="clean_models/${clean_name}"
data_dir="data/"
unlabeled_targets_dir="unlabeled_targets"
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
init_lr=0.1

flags=--add_aux_labels
name="adv-${distance}-${model}-take_frac=${take_fraction}-take_seed=${take_amount_seed}-unw=${unlabeled_natural_weight}-urw=${unlabeled_robust_weight}-seed=${seed}-beta=${beta}-max_rot=${max_rot}-max_trans=${max_trans}"
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
    --distance ${distance} \
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
    --num-steps ${num_steps} \
    --max_rot ${max_rot} \
    --max_trans ${max_trans} \
    --lr ${init_lr} \

# Running final attack 

echo Test with final attack

num_steps=40
step_size=0.01
num_restarts=5

checkpoint_path="${output_dir}/checkpoint-epoch${epochs}.pt"
output_suffix="max_rot=${max_rot}-max_trans=${max_trans}-steps=${num_steps}-ss=${step_size}"
python $DIR/code/spatial_attack_cifar10.py \
       --model-path ${checkpoint_path} \
       --model ${model} \
       --num-steps ${num_steps} \
       --step-size ${step_size} \
       --output-suffix ${output_suffix} \
       --max-rot ${max_rot} \
       --max-trans ${max_trans}
