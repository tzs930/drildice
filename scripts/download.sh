#!/bin/bash
# When a condor job is on hold, use 'condor_q -analyze {cid}.{pid}'
export DISABLE_TQDM=TRUE
rm -rf /tmp/d4rl; mkdir /tmp/d4rl ;
source ~/.bashrc ;
cd /ext2/syseo/hsicbc ;
D4RL_DATASET_DIR=/tmp/d4rl /home/syseo/anaconda3/envs/d4rlpy38/bin/python download_d4rl_dataset.py