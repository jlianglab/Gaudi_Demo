#!/bin/bash
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 3-00:00:00
#SBATCH -c 5
#SBATCH --mem=48G

APPTAINER_CMD=(../../gaudi-apptainer.sh exec)
export PT_HPU_LAZY_MODE=1

"${APPTAINER_CMD[@]}" python /workspace/BenchmarkTransferLearning/main_segmentation.py \
--data_set SIIM_PNE \
--backbone swin_base \
--arch swin_upernet \
--init imagenet_1k \
--proxy_dir <path_to_model> \
--config /workspace/segmentation_config.yaml \
--save_dir <output_dir> \
--train_num_workers 4 \
--test_num_workers 4 \
--run 1 \
--device hpu