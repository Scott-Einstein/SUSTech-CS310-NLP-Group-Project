#!/bin/bash
#SBATCH -o job.%j.out                 # 输出日志
#SBATCH --partition=a100             # 分区队列
#SBATCH --qos=a100                   # QOS
#SBATCH -J NLPfineTuningTask                # 作业名
#SBATCH --nodes=1                     # 节点数
#SBATCH --ntasks-per-node=1           # 每节点任务数
#SBATCH --gres=gpu:1                  # 申请1张GPU
source activate NLP_env
python run.py
