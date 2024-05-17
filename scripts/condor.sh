#!/bin/bash
# When a condor job is on hold, use 'condor_q -analyze {cid}.{pid}'
# echo "hostname: `hostname`"
# echo "PATH=$PATH"
# echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
export DISABLE_TQDM=TRUE
source ~/.bashrc ;
conda activate d4rlpy38 ;
cd /ext2/syseo/hsicbc ;
WANDB_SERVICE_WAIT=3000 WANDB_DIR=/tmp D4RL_DATASET_DIR=/ext2/syseo/hsicbc/dataset  ~/anaconda3/envs/d4rlpy38/bin/python  train_imbalance.py --pid=$1