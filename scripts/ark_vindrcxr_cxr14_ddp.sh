#!/bin/bash
#SBATCH -p gaudi
#SBATCH -q public
#SBATCH -N 1
#SBATCH -G 2
#SBATCH -t 7-00:00:00
#SBATCH -c 10
#SBATCH --mem=96G

APPTAINER_CMD=(./gaudi-apptainer.sh exec)
export PT_HPU_LAZY_MODE=1
export OMP_NUM_THREADS=1

"${APPTAINER_CMD[@]}" python -m torch.distributed.run --nproc_per_node 2 /workspace/Ark_Plus/Pretraining/main_ark.py --output_dir /scratch/echang32/Ark_Outputs/ddp --data_set VinDrCXR --data_set ChestXray14 --device hpu --opt sgd --warmup-epochs 20  --lr 0.3  --batch_size 100 --workers 4 --model swin_base --init imagenet  --pretrain_epochs 200  --test_epoch 1 --pretrained_weights https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth --momentum_teacher 0.9  --projector_features 1376