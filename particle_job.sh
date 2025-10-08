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

propfile_arg="$1"
shift
echo $propfile_arg
propfile_addr="${propfile_arg#--propfile_addr=}"

uv_args=("$@")
echo "${uv_args[@]}"

# Discretization
cd ../../Heat_Pump/
source ./.venv/bin/activate
cd nfer_format_compatibility
echo "The actual script started running at: $(date)"
cp $infile_addr "$TMPDIR"/$(basename "$infile_addr")
cp $propfile_addr "$TMPDIR"/$(basename "$propfile_addr")
uv run event-trace-extraction-by-CPD "$TMPDIR"/$(basename "$infile_addr") "$TMPDIR"/$(basename "$propfile_addr") --base-dir "$TMPDIR/CPD_Analysis_Outputs" --data-timezone "Europe/Paris" --split-time "08:00" --alg-params "${uv_args[@]}"
deactivate

# Automata Learning
module load java/21.0.1
hpm_executable=build/distributions/heat-pump-mining-0.0.0-14-ad1ef39-SNAPSHOT/
EventTracePath="$TMPDIR/CPD_Analysis_Outputs/event_traces.zip"
# cp "$EventTracePath" "./CPD_Analysis_Outputs/event_traces_${SLURM_JOB_ID}.zip"

cd ../../hpm_automata_learning/heat-pump-mining/
exec_path=build/distributions/heat-pump-mining-0.0.0-14-ad1ef39-SNAPSHOT/bin/heat-pump-mining
echo "The actual script started running at: $(date)"
echo "Starting convert-format..."
$exec_path convert-format -i $EventTracePath -o "$TMPDIR/nibe.json.zst"
echo "Starting select-and-merge..."
# $exec_path select-and-merge -i "$TMPDIR/nibe.json.zst" -o "$TMPDIR/filtered.json.zst" -e defrost baud_rate -s defrost -a
$exec_path select-and-merge -i "$TMPDIR/nibe.json.zst" -o "$TMPDIR/filtered.json.zst" -e flow_temperature return_flow_temperature hc_zone1 set_flow_zone_1 -s set_flow_zone_1 -a
echo "Starting learn..."
$exec_path learn -i "$TMPDIR/filtered.json.zst" -o "$TMPDIR/nibe.dot"
echo "Starting validate..."
$exec_path validate -a "$TMPDIR/nibe.dot" -i "$TMPDIR/filtered.json.zst"

# Don't copy the files yet, you need to use them in the next step (but ./outputs does not exist;
# it must copy to the parent process's ./outputs, must come to this script as an input argument)
# cp -r "$TMPDIR/CPD_Analysis_Outputs/" ./outputs/

echo "Particle job ended at: $(date)"
