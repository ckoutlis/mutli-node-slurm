* Create the environment with:
```
conda create -n mns python=3.10 -y
conda activate mns
conda install pytorch==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

* Run any of the scripts in `slurm/sh`:
```
# To train on one machine with two GPUs:
sbatch slurm/sh/OneNodeTwoGPUs.sh
```