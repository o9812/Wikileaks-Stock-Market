#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=countrycl
#SBATCH --output=cc_slurm_%j.out
#SBATCH --error=cc_slurm_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:k80:1

#SBATCH --mail-type=end  # email me when the job ends

# Change the home directory
cd /home/ypl238/ml_project/
python3 -u RanFrst_classfy.py 100 ./data_country_ng/ country_100_class_ng -country
