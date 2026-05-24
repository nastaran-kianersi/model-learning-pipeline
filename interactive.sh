#!/bin/bash

export PYTHONUNBUFFERED=TRUE

source ~/HPM_Pipeline/hpm_pipeline/configs.env

infile_raw_path=$EXEC_TRACE_FILE
configs_dir_raw_path=$PROP_CONFIG

copy_name="configs_${SLURM_JOB_ID}.env"
cp ~/HPM_Pipeline/hpm_pipeline/configs.env "$copy_name"
echo "Wrote copy of configs.env to: $copy_name"
cp -r $configs_dir_raw_path ~/"configs_${SLURM_JOB_ID}"

# convert to absolute path
input_addr=$(cd -- "$(dirname -- "$infile_raw_path")" && pwd)/$(basename -- "$infile_raw_path")
echo "Resolved input file/folder path: $input_addr"
#configs_dir=$(cd -- "$(dirname -- "$configs_dir_raw_path")" && pwd)
configs_dir=$(cd $configs_dir_raw_path; pwd)
#echo "Resolved configs dir: $configs_dir"
echo "$SLURM_JOB_ID"
echo "Pipeline job started at: $(date)"


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
pip install python-dotenv --user
#module load scipy-stack/2023b matplotlib/3.7.2
#source venv/bin/activate
echo "The actual script started running at: $(date)"
python3 main.py "$input_addr" "$configs_dir"
#deactivate
echo "Pipeline job ended at: $(date)"
