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

"${APPTAINER_CMD[@]}" python /workspace/BenchmarkTransferLearning/main_classification.py \
--data_set VinDrCXR \
--init imagenet_1k \
--proxy_dir <path_to_model> \
--config /workspace/classification_config.yaml \
--model swin_base \
--save_dir <output_dir> \
--trial 3 \
--num_class 6 \
--workers 4 \
--device hpu