# Entailment Classifier

## Singularity Image on SLURM

We must use
[Singularity](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda)
in order to first create a container with all of the dependencies which can then be used when submitting the 
SLURM batch job.

Assuming the container is already set up (i.e. contains all of the dependencies per the above link):

```
cd /scratch/nn1331/entailment

# Following the above official documentation
srun --cpus-per-task=2 --mem=10GB --time=04:00:00 --pty /bin/bash

# wait to be assigned a node

singularity exec --overlay overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash

source /ext3/env.sh
# activate the environment
```

This can be used for updating the dependencies within a Singularity Image as well!

Specifically, the following dependencies should be installed in the conatiner:
`pip3 install torch datasets evaluate numpy transformers scikit-learn hydra-core omegaconf bitarray sacrebleu`


## On SLURM

```
# Submit job
sbatch entailment_classifier.sbatch

# Monitor job status
squeue -u $USER

# View job output
cat python-entailment-classifier.out
```

## Legacy instructions

### Installation

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Run

```
./entailment_classifier.py
```

### Conda

```
module load anaconda3/2020.07
conda env create -f environment.yml
conda activate entailment
```

The above will not work since it will cause the local environment to run out of memory.

