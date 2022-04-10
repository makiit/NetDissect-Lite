#!/usr/bin/bash

module purge
module load slurm # Simple Linux Utility for Resource Management (SLURM) batch environment

# gpu
srun --partition=gpu-debug --pty --account=ddp390 --ntasks-per-node=10 --nodes=1 --mem=96G --gpus=1 -t 00:30:00 --wait=0 --export=ALL /bin/bash
