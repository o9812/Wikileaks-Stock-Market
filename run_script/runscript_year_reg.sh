#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=year_rg
#SBATCH --output= yr_slurm_%j.out
#SBATCH --error= yr_slurm_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:k80:1

#SBATCH --mail-type=end  # email me when the job ends

# Change the home directory
cd /home/ypl238/ml_project/
python3 -u RanFrst_regres_final.py 100 ./data_year/ year_100 -year
