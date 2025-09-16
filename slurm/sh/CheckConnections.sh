#!/bin/bash
#SBATCH --qos=test
#SBATCH --job-name=connect
#SBATCH --nodes=2
#SBATCH --gpus-per-node=rtx_4090:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/out/connect.out
#SBATCH --error=slurm/out/connect.err

export PYTHONPATH="$HOME/code/mutli-node-slurm"
cd $HOME/code/mutli-node-slurm
source $HOME/.bashrc
source $HOME/anaconda3/bin/activate
conda activate mns

echo "Starting non-interactive network debugging script..."
echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "=================================================="

MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_ADDR=$MASTER_ADDR
echo "MASTER_ADDR: $MASTER_ADDR"
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
echo "MASTER_PORT: $MASTER_PORT"
echo "=================================================="

NODES=($(scontrol show hostnames $SLURM_NODELIST))

echo "Running ping tests from master node ($MASTER_ADDR) to all other nodes..."
for node in "${NODES[@]}"; do
    if [[ "$node" != "$MASTER_ADDR" ]]; then
        echo "Pinging $node..."
        ping -c 3 $node
        if [ $? -eq 0 ]; then
            echo "Ping to $node successful."
        else
            echo "Ping to $node FAILED."
        fi
    fi
done
echo "=================================================="

echo "Running netcat (nc) tests for TCP connectivity on port $MASTER_PORT..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    bash -c '
        current_hostname=$(hostname)
        if [[ "$SLURM_NODEID" -eq "0" ]]; then
            # The master node will listen on the port
            echo "Master node ($current_hostname) listening on port $MASTER_PORT..."
            nc -l -p $MASTER_PORT > /dev/null &
            # Wait for 5 seconds for other nodes to connect
            sleep 5
            echo "Master node nc listener terminated."
        else
            # All other nodes will try to connect
            echo "Node ($current_hostname) connecting to master ($MASTER_ADDR) on port $MASTER_PORT..."
            nc -zv $MASTER_ADDR $MASTER_PORT
            if [ $? -eq 0 ]; then
                echo "Connection from $current_hostname to $MASTER_ADDR successful."
            else
                echo "Connection from $current_hostname to $MASTER_ADDR FAILED."
            fi
        fi
    '
echo "=================================================="

export NCCL_SOCKET_IFNAME=$(ip addr show | awk '/inet 10.*brd/{print $NF}')
export NCCL_DEBUG=INFO

echo "Attempting to run the PyTorch distributed job..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
    torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $HOME/code/mutli-node-slurm/train.py