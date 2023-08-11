#!/usr/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=test_job.%j.out
#SBATCH --error=test_job.%j.err
#SBATCH -p gpu
#SBATCH -G 1

ml py-pytorch/1.8.1_py39
python3 -u multitask_classifier.py --option pretrain --epochs 10 --lr 1e-3 --batch_size 32 --use_gpu
