#!/bin/bash
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 3-00:00:00
#SBATCH -c 8
#SBATCH --mem=24G

APPTAINER_CMD=(../gaudi-apptainer.sh exec)

export PT_HPU_LAZY_MODE=1
export OMP_NUM_THREADS=1 

"${APPTAINER_CMD[@]}" python -m torch.distributed.run --nproc_per_node=1 --master_port 28500 /workspace/BenchmarkTransformers/main_classification.py --model swin_base --init ImageNet_21k --data_set ChestXray14 \
--output_dir <output_dir> --config /workspace/config.yaml --device hpu --trial 3 \
--epochs 200 --batch_size 64