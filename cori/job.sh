#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -J fisher
#SBATCH --mail-user=a.kalaja@rug.nl
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00
#SBATCH --mem=50GB

#OpenMP settings:
export OMP_NUM_THREADS=32
###export OMP_PLACES=threads
###export OMP_PROC_BIND=true


#run the application:
module load openmpi
conda activate albaenv
srun python /global/cscratch1/sd/akalaja/main.py
