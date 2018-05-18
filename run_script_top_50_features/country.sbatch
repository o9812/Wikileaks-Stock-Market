#!/bin/bash
#SBATCH --job-name=all-country
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=152:00:00

#module purge
#module load jupyter-kernels/py3.5

#module load tensorflow/python2.7/20170707
#module load keras/2.0.2
#module load scikit-learn/intel/0.18.1


#module load torch/gnu
#module load torchvision/0.1.8
#module load pytorch/python3.6/0.2.0_3
source /home/jw4937/py3.6.3/bin/activate

python -i /home/jw4937/Machine_learning/Wikileaks-Stock-Market/RanFrst_regres.py 30 /home/jw4937/Machine_learning/Data_Folder/data_all_country_neg/ 0518_country -country