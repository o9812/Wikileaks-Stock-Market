#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=training
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=2GB
#SBATCH --gres=gpu:k80:1

#SBATCH --mail-type=end  # email me when the job ends

# Change the home directory
cd /home/ypl238/ml_project/
#module load pytorch/python3.6/0.3.0_4
#module load scipy/intel/0.19.1
#module load scikit-learn/intel/0.18.1
#module load numpy/python3.6/intel/1.14.0
python3 -u RanFrst_classfy.py 100 ./data_year/ year_100_class -year
