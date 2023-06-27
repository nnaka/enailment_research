#!/bin/bash 
#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=2:00:00
#SBATCH --mem=2GB
#SBATCH --gres=gpu
#SBATCH --job-name=python-entailment-classifier
#SBATCH --output=python-entailment-classifier.out

module purge
#module load python/intel/3.8.6
#
#python3 -m venv venv
#source venv/bin/activate
#pip3 install -r requirements.txt
#./entailment_classifier.py

singularity exec --nv \
	    --overlay /scratch/nn1331/entailment/entailment.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python entailment_classifier.py"
