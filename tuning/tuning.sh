#!/bin/bash


#SBATCH --time=4:00:00
#SBATCH --mem=75G
#SBATCH --cpus-per-task=32
#SBATCH --account=def-haricots

module load python
source tensorflow/bin/activate

python TransformerOptimInteractive.py
