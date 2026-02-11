#!/bin/bash
#SBATCH --mem=2G
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
##SBATCH --exclusive
#SBATCH -t 0-23:00:00
#SBATCH --output=job_%A.out
#SBATCH --error=job_%A.err
##SBATCH --mail-user=nastaran.kianersi@queensu.ca
##SBATCH --mail-type=ALL
#SBATCH --qos=privileged
#SBATCH --partition=reserved

export PYTHONUNBUFFERED=TRUE

# to get the absolute path from the full path:
#readlink -f $1 # doesn't work
# check for input argument or use default
if [ -n "$1" ]; then
    infile_raw_path="$1"
else
    infile_raw_path="./Heat_Pump/nfer_format_compatibility/mitsubishi_12_december2023+jan2024_method1.events"
    # infile_raw_path="/global/home/hpc5649/LearnerInputSignals/buckboost_close_vrefs_2/buckboost_close_vrefs_2.csv"   #"./c0l13r8vref18nominal2_10repeats.csv"
fi

if [ -n "$2" ]; then
    configs_dir_raw_path="$2"
else
    configs_dir_raw_path="./HPM_Pipeline/hpm_pipeline/tests/configs_2"
    # configs_dir_raw_path="./HPM_Pipeline/hpm_pipeline/tests/configs_3"
fi

# convert to absolute path
input_addr=$(cd -- "$(dirname -- "$infile_raw_path")" && pwd)/$(basename -- "$infile_raw_path")
#echo "Resolved input file path: $input_addr"
#configs_dir=$(cd -- "$(dirname -- "$configs_dir_raw_path")" && pwd)
configs_dir=$(cd $configs_dir_raw_path; pwd)
#echo "Resolved configs dir: $configs_dir"
echo "$SLURM_JOB_ID"
echo "Pipeline job started at: $(date)"

# clear scratch memory and make sure it's been cleaned up
# cd /global/scratch/hpc5649/
# rm -rf *
# rm -rf .*
# if [ -z "$( ls -A '/global/scratch/hpc5649/' )" ]; then
#    echo "Empty"
# else
#    echo "Not Empty"
# fi
# cd ~

# chmod -R u+w /global/scratch/hpc5649/
chmod u+r /global/scratch/hpc5649/
chmod u+w /global/scratch/hpc5649/

if [ -z "$( ls -A '/global/scratch/hpc5649/' )" ]; then
   echo "Before clearing: empty"
else
   echo "Before clearing: not empty"
   rm -rf /global/scratch/hpc5649/*   # removes everything including hidden ones

#   dir="/global/scratch/hpc5649/"
#    find "$dir" -type l -delete               # Remove ALL symlinks (dangling or not)
#    find "$dir" -mindepth 1 -delete 2>/dev/null  # Remove everything
fi

# shopt -s dotglob                   # include dotfiles in globs
# rm -rf /global/scratch/hpc5649/*   # removes everything including hidden ones

if [ -z "$( ls -A '/global/scratch/hpc5649/' )" ]; then
   echo "After clearing: empty"
else
   echo "After clearing: not empty"
fi

cd HPM_Pipeline/hpm_pipeline
chmod +x *.sh
#module load StdEnv/2023 python/3.12.4 scipy-stack/2024a
module load StdEnv/2023 python/3.10.13 scipy-stack/2023b
#module load scipy-stack/2023b matplotlib/3.7.2
#source venv/bin/activate
echo "The actual script started running at: $(date)"
python3 main.py "$input_addr" "$configs_dir"
#deactivate
echo "Pipeline job ended at: $(date)"
