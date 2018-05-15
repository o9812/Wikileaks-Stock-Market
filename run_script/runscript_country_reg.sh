#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=countryreg
#SBATCH --output=cr_slurm_%j.out
#SBATCH --error=cr_slurm_%j.err
#SBATCH --time=96:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:k80:1

#SBATCH --mail-type=end  # email me when the job ends

# Change the home directory
cd /home/ypl238/Wikileaks-Stock-Market/

python3 -u RanFrst_regres.py 1 ./data_all_country_neg/ country_neg_1 -country
