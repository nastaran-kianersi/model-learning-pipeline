#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=16384
#SBATCH -t 0-01:00:00
#SBATCH --output=job_%A.out
#SBATCH --error=job_%A.err
#SBATCH --qos=privileged
#SBATCH --partition=reserved

# Disable Python output buffering to ensure real-time logging
export PYTHONUNBUFFERED=TRUE

# commands
echo "Particle job started at: $(date)"

# Construct the uv command arguments
file_arg="$1"
shift
echo $file_arg
file_addr="${file_arg#--file_addr=}"
uv_args=("$@")
echo "${uv_args[@]}"

cd ../../Heat_Pump/
source ./.venv/bin/activate
cd nfer_format_compatibility
echo "The actual script started running at: $(date)"
cp $file_addr "$TMPDIR"/$(basename "$file_addr")
uv run event-trace-extraction-by-CPD "$TMPDIR"/$(basename "$file_addr") --base-dir "$TMPDIR/CPD_Analysis_Outputs" --data-timezone "Europe/Paris" --split-time "08:00" --alg-params "${uv_args[@]}"

# Don't copy the files yet, you need to use them in the next step (but ./outputs does not exist;
# it must copy to the parent process's ./outputs, must come to this script as an input argument)
# cp -r "$TMPDIR/CPD_Analysis_Outputs/" ./outputs/

deactivate

# Zip the event traces
# TODO
# Find the newest folder in CPD_Analysis_Outputs and zip it
# echo "Zipping newest folder in CPD_Analysis_Outputs..."
# newest_dir=$(find ./CPD_Analysis_Outputs/ -mindepth 1 -maxdepth 1 -type d -exec stat -c "%Y %n" {} + | sort -nr | head -1 | cut -d' ' -f2-)
# zipname="event_traces_$(basename "$newest_dir").zip"
# zip -r "$zipname" "$newest_dir"

module load java/17.0.6

# Make the path dynamic
# EventTracePath="/global/home/hpc5649/Heat_Pump/nfer_format_compatibility/CPD_Analysis_Outputs/alg_PELT_kernel_rbf_gamma_1_min_size_300_penalty_50/event_traces/NibeData_EventTraces_0.zip"
EventTracePath="$TMPDIR/CPD_Analysis_Outputs/event_traces.zip"

#./gradlew application
# how to run that
# unpack the zip and get all the executables (ls build)
# Read files from $TMPDIR
cd ../../hpm_automata_learning/heat-pump-mining/
echo "The actual script started running at: $(date)"
echo "Starting convert-format..."
./gradlew run --args "convert-format -i \"$EventTracePath\" -o \"$TMPDIR/nibe.json.zst\""
echo "Starting select-and-merge..."
./gradlew run --args "select-and-merge -i \"$TMPDIR/nibe.json.zst\" -o \"$TMPDIR/filtered.json.zst\" -e defrost baud_rate -s defrost -a"
echo "Starting learn..."
./gradlew run --args "learn -i \"$TMPDIR/filtered.json.zst\" -o \"$TMPDIR/nibe.dot\""
echo "Starting validate..."
./gradlew run --args "validate -a \"$TMPDIR/nibe.dot\" -i \"$TMPDIR/filtered.json.zst\""

# Get the output of "validate" and calculate the sum (or mean) of scores and output the results

echo "Particle job ended at: $(date)"
