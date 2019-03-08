#!/bin/bash
#SBATCH -J my_job             # Job name
#SBATCH -o my_job.o%j         # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu                # Queue name
#SBATCH -N 1                  # Total number of nodes requested (20 cores/node) --> note: >1 only works with Launcher or MPI
#SBATCH -n 1                  # Total number of mpi tasks requested
#SBATCH -t 12:00:00           # Run time (hh:mm:ss) - 1.5 hours
#SBATCH -A SD2E-Community     # <-- Allocation name to charge job against

module load tacc-singularity/2.6.0

singularity exec --nv ${WORKSPACE}/singularity_cache/tacc-maverick-ml-latest.simg python3 ${WORKSPACE}/scripts/perovskite_test_harness.py
