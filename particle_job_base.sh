#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 0-00:45:00
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH --wait

# use sbatch --unbuffered to get the outputs in real-time (but it will bread the Pipeline)

# Disable Python output buffering to ensure real-time logging
export PYTHONUNBUFFERED=TRUE

echo "Particle job started at: $(date)"

# Construct the uv command arguments
infile_arg="$1"
shift
echo $infile_arg
infile_addr="${infile_arg#--infile_addr=}"

config_dir_arg="$1"
shift
echo $config_dir_arg
config_dir="${config_dir_arg#--config_dir=}"

uv_args=("$@")
echo "${uv_args[@]}"

# Discretization
cd ../../Heat_Pump/
source ./.venv/bin/activate
cd nfer_format_compatibility
echo "The actual script started running at: $(date)"
cp $infile_addr "$TMPDIR"/$(basename "$infile_addr")
cp "$config_dir/optimizing_props.prop" "$TMPDIR"/optimizing_props.prop
uv run event-trace-extraction-by-CPD "$TMPDIR"/$(basename "$infile_addr") "$TMPDIR"/optimizing_props.prop --base-dir "$TMPDIR/CPD_Analysis_Outputs" --data-timezone "Europe/Paris" --split-time "08:00" --alg-params "${uv_args[@]}"
deactivate

# Don't copy the files yet, you need to use them in the next step (but ./outputs does not exist;
# it must copy to the parent process's ./outputs, must come to this script as an input argument)
# cp -r "$TMPDIR/CPD_Analysis_Outputs/" ./outputs/

echo "Particle job ended at: $(date)"
