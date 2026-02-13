#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=15G
#SBATCH -t 0-00:15:00
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


# Discretization
cd ../../Heat_Pump/
source ./.venv/bin/activate
cd nfer_format_compatibility
echo "The actual script started running at: $(date)"
# cp $infile_addr "$TMPDIR"/$(basename "$infile_addr")
if [ -d "$infile_addr" ]; then
  cp -r "$infile_addr" "$TMPDIR"/
else
  cp "$infile_addr" "$TMPDIR"/
fi
cp "$config_dir/onetime_props.prop" "$TMPDIR"/onetime_props.prop
# --split-time "08:00"
# OutputPath=$(uv run event-trace-extraction-by-CPD "$TMPDIR"/$(basename "$infile_addr") "$TMPDIR"/onetime_props.prop --base-dir "$TMPDIR/CPD_Analysis_Outputs" --data-timezone "Europe/Paris" | tail -n 1)
uv run event-trace-extraction-by-CPD \
    "$TMPDIR"/$(basename "$infile_addr") \
    "$TMPDIR"/onetime_props.prop \
    --split-time "08:00" \
    --base-dir "$TMPDIR/CPD_Analysis_Outputs" \
    --data-timezone "Europe/Paris" \
    --enable-train-test-split --train-ratio=0.75
deactivate

# create a folder in scratch mem called event_traces_ontime and copy the content of OutputPath there
# Read the file into an array (one element per line)
mapfile -t paths < /tmp/output_path.txt

# Process first line (always exists)
OutputPath="${paths[0]}"
echo "Train discretization output path: $OutputPath"

SCRATCH_DIR="/global/scratch/hpc5649"
EventTracePath="$SCRATCH_DIR/train_event_traces_onetime"
echo "Train event trace path: $EventTracePath"
mkdir -p "$EventTracePath"
cp -r "$OutputPath"/* "$EventTracePath"

# Process second line if it exists
if [ ${#paths[@]} -eq 2 ]; then
    OutputPath2="${paths[1]}"
    echo "Test discretization output path: $OutputPath2"
    
    EventTracePath2="$SCRATCH_DIR/test_event_traces_onetime"
    echo "Test event trace path: $EventTracePath2"
    mkdir -p "$EventTracePath2"
    cp -r "$OutputPath2"/* "$EventTracePath2"
fi

# mkdir -p ./outputs/event_traces_onetime # prepare the target location
# cp -R "$OutputPath"/. ./outputs/event_traces_onetime # copy only the contents

echo "Particle job ended at: $(date)"
