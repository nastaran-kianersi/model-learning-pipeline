#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 0-00:45:00
#SBATCH --output=job_%A.out
#SBATCH --error=job_%A.err
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH --wait
#SBATCHhhhh --unbuffered

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

# Discretization
cd ../../Heat_Pump/
source ./.venv/bin/activate
cd nfer_format_compatibility
echo "The actual script started running at: $(date)"
cp $file_addr "$TMPDIR"/$(basename "$file_addr")
uv run event-trace-extraction-by-CPD "$TMPDIR"/$(basename "$file_addr") --base-dir "$TMPDIR/CPD_Analysis_Outputs" --data-timezone "Europe/Paris" --split-time "08:00" --alg-params "${uv_args[@]}"
deactivate

# Automata Learning
module load java/17.0.6
EventTracePath="$TMPDIR/CPD_Analysis_Outputs/event_traces.zip"

#./gradlew application => NO. distZip
# unpack the zip and get all the executables (ls build)
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

#
# TODO: Process the output
# Get the output of "validate" and calculate the sum (or mean) of scores and output the results

# Don't copy the files yet, you need to use them in the next step (but ./outputs does not exist;
# it must copy to the parent process's ./outputs, must come to this script as an input argument)
# cp -r "$TMPDIR/CPD_Analysis_Outputs/" ./outputs/

echo "Particle job ended at: $(date)"
