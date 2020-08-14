#!/bin/bash

set -x

mem=32g
gpus=2
cpus=2
CODALAB_WKSHT=YOUR_WKSHT_HERE

bash sample_sizes.sh
bash run_pgat.sh
cl run -m -n table1-selftrain -w $CODALAB_WKSHT --request-docker-image sangxie513/robust-tradeoff-codalab-gpu --request-gpus ${gpus} --request-memory ${mem} --request-cpus ${cpus} --request-queue tag=nlp data:data code:code train_pipeline_pgat.sh:train_pipeline_pgat.sh run_for_sample_size.sh:run_for_sample_size.sh train_cleanmodel.sh:train_cleanmodel.sh train_pipeline_rotations.sh:train_pipeline_rotations.sh train_trades_pipeline.sh:train_trades_pipeline.sh run_selftrain.sh:run_selftrain.sh --- bash run_selftrain.sh
cl run -m -n table1-rotations -w $CODALAB_WKSHT --request-docker-image sangxie513/robust-tradeoff-codalab-gpu --request-gpus ${gpus} --request-memory ${mem} --request-cpus ${cpus} --request-queue tag=nlp data:data code:code train_pipeline.sh:train_pipeline.sh run_for_sample_size.sh:run_for_sample_size.sh train_cleanmodel.sh:train_cleanmodel.sh train_pipeline_rotations.sh:train_pipeline_rotations.sh train_trades_pipeline.sh:train_trades_pipeline.sh run_rotations.sh:run_rotations.sh --- bash run_rotations.sh
