#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH -p general
#SBATCH -q private
#SBATCH --cpus-per-task=10

module load mamba/latest
chmod +x gaudi-apptainer.sh
./gaudi-apptainer.sh -i gaudi-apptainer.sif -d gaudi-apptainer.def build

chmod +x cuda-apptainer.sh
./cuda-apptainer.sh -i cuda-apptainer.sif -d cuda-apptainer.def build