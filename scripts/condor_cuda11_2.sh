#!/bin/bash
# When a condor job is on hold, use 'condor_q -analyze {cid}.{pid}'
# echo "hostname: `hostname`"
# echo "PATH=$PATH"
# echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
export DISABLE_TQDM=TRUE
source ~/.bashrc ;
conda activate d4rlcuda11 ;
cd /ext2/syseo/hsicbc ;
WANDB_SERVICE_WAIT=300 WANDB_DIR=/tmp /home/syseo/anaconda3/envs/d4rlcuda11/bin/python train2.py --pid=$1