#!/bin/bash
#SBATCH --qos=test
#SBATCH --job-name=1n2g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx_4080:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/out/1n2g.out
#SBATCH --error=slurm/out/1n2g.err

export PYTHONPATH="$HOME/code/mutli-node-slurm"
cd $HOME/code/mutli-node-slurm
source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate mns

echo "SLURM_JOB_NAME: $SLURM_JOB_NAME"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "=================================================="

MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
echo "MASTER_ADDR: $MASTER_ADDR"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "MASTER_PORT: $MASTER_PORT"
echo "=================================================="

NUM_GPUS=$SLURM_GPUS_PER_NODE
if [[ $SLURM_GPUS_PER_NODE == *:* ]]; then
    NUM_GPUS=$(echo $SLURM_GPUS_PER_NODE | awk -F: '{print $NF}')
fi
echo "NUM_GPUS_PER_NODE: $NUM_GPUS"

srun --nodes=$SLURM_NNODES --ntasks-per-node=$SLURM_NTASKS_PER_NODE \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $HOME/code/mutli-node-slurm/train.py