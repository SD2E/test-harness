#!/bin/bash
#SBATCH -J Parametric          # Job name
#SBATCH -N 8                   # Total number of nodes (16 cores/node)
#SBATCH -n 8                   # Total number of tasks
#SBATCH -p gpu                 # Queue name
#SBATCH -o Parametric.o%j      # Name of stdout output file (%j expands to jobid)
#SBATCH -t 10:00:00            # Run time (hh:mm:ss)
#SBATCH -A SD2E-Community

#------------------General Options---------------------
module load launcher
export TACC_LAUNCHER_PPN=1
export EXECUTABLE=$TACC_LAUNCHER_DIR/init_launcher
export WORKDIR=.
export CONTROL_FILE=hugh_jobfile
export TACC_LAUNCHER_SCHED=interleaved

module load tacc-singularity/2.6.0
#export MYIMG="/work/05260/hamed/singularity_cache/test-harness-3.2.simg"
#export MYCMD="singularity exec --nv ${MYIMG}"


cd $WORKDIR/
echo " WORKING DIR:   $WORKDIR/"

$TACC_LAUNCHER_DIR/paramrun SLURM $EXECUTABLE $WORKDIR $CONTROL_FILE $PHI_WORKDIR $PHI_CONTROL_FILE

echo " "
echo " Parametric Job Complete"
echo " "
