#!/bin/bash
#SBATCH -J my_job             # Job name
#SBATCH -o my_job.o%j         # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu                # Queue name
#SBATCH -N 1                  # Total number of nodes requested (20 cores/node) --> note: >1 only works with Launcher or MPI
#SBATCH -n 1                  # Total number of mpi tasks requested
#SBATCH -t 12:00:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -A SD2E-Community     # <-- Allocation name to charge job against

module load tacc-singularity/2.6.0
module load remora

MYIMG="/work/05260/hamed/singularity_cache/test-harness-3.2.simg"
MYCMD="singularity exec --nv ${MYIMG}"

remora ${MYCMD} python3 /work/05260/hamed/maverick/protein-design/test_harness/scripts/hamed_test_scripts/loo_memory_test.py
