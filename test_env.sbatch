#!/bin/bash
#SBATCH --job-name=gh_256_16_stride2
#SBATCH --account=pawsey1001-gpu
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=12:10:00
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

module load rocm/5.6.1
source /scratch/pawsey1001/mliu2/sf/SlowFast/env/bin/activate
cd /scratch/pawsey1001/mliu2/sf/SlowFast

# 手动打印环境变量 & 当前路径
# echo "[INFO] PWD: $(pwd)"
# echo "[INFO] Python version: $(python --version)"
# echo "[INFO] Listing config file:"
# ls -lh configs/GreatHits/greathits.yaml
# export MASTER_ADDR=$(hostname -I | awk '{print $1}')  # 如 172.23.0.13
# export MASTER_PORT=29501

# 打印当前时间戳
date
python -c "import torch; print("gpu数量"); print(torch.cuda.device_count())"

# 启动训练（无端口随机化，先稳定运行）--init_method tcp://10.253.128.17:49717
PYTHONPATH=$(pwd) python tools/run_net.py --cfg configs/GreatHits/gh_256_16_stride2.yaml

# 打印结束时间戳
date
