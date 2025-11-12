#!/bin/bash
#SBATCH -o /global/home/hpc5649/HPM_Pipeline/hpm_pipeline/outputs/run_2025-11-10_at_17-31-57/iter_4/particle_4/job_%A.out
#SBATCH -e /global/home/hpc5649/HPM_Pipeline/hpm_pipeline/outputs/run_2025-11-10_at_17-31-57/iter_4/particle_4/job_%A.err
##SBATCH --output=/dev/null
##SBATCH --error=/dev/null
#SBATCH -c 1
#SBATCH --mem=50
#SBATCH -t 0-00:05:00
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH --wait
# use sbatch --unbuffered to get the outputs in real-time (but it will bread the Pipeline)


# Usage:
#   ./merge_event_traces.sh 1 3 26
# This merges event_traces_1, event_traces_3, and event_traces_26 into event_traces/

# Check that at least two arguments are given
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <out_filename> <id1> [id2 id3 ...]"
    exit 1
fi

infile_arg="$1"
shift
echo $infile_arg
EventTraceFoldername="${infile_arg#--EventTraceFoldername=}"

SCRATCH_DIR="/global/scratch/hpc5649"
EventTracePath="$SCRATCH_DIR/$EventTraceFoldername"

# Create the unified directory
mkdir -p "$EventTracePath/split_with_same_suffix/continuous" 
mkdir -p "$EventTracePath/split_with_same_suffix/discrete"

# Loop through provided suffixes
for i in "$@"; do
    dir="$SCRATCH_DIR/event_traces_${i}"

    if [[ ! -d "$dir" ]]; then
        echo "Directory $dir does not exist, skipping."
        continue
    fi

    # echo "Merging from $dir ..."

    # Copy continuous vars
    if [[ -d "$dir/split_with_same_suffix/continuous" ]]; then
        cp -r "$dir/split_with_same_suffix/continuous/"* "$EventTracePath"/split_with_same_suffix/continuous/ 2>/dev/null
    fi

    # Copy discrete vars
    if [[ -d "$dir/split_with_same_suffix/discrete" ]]; then
        cp -r "$dir/split_with_same_suffix/discrete/"* "$EventTracePath"/split_with_same_suffix/discrete/ 2>/dev/null
    fi
done

# Zip the merged folder
cd "$(dirname "$EventTracePath")"
zip -r "$(basename "$EventTracePath").zip" "$(basename "$EventTracePath")"
# zip -r "${EventTracePath}.zip" "$EventTracePath"
rm -rf "$EventTracePath"
echo "Event trace path: ${EventTracePath}.zip" #\"$EventTracePath\".zip"
