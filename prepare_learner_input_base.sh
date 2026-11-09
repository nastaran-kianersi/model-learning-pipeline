##SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH -c 1
#SBATCH --mem=100M
#SBATCH -t 0-00:15:00
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
TrainEventTracePath="$SCRATCH_DIR/train_$EventTraceFoldername"
TestEventTracePath="$SCRATCH_DIR/test_$EventTraceFoldername"

for i in "$@"; do
    for prefix in "train" "test"; do
        dir="$SCRATCH_DIR/${prefix}_event_traces_${i}"
        # Discover the split directory name dynamically
        if [[ -d "$dir" ]]; then
            SplitDirName=$(ls -d "$dir"/*/ 2>/dev/null | head -1 | xargs basename)
            break
        else
            echo "Directory $dir does not exist. Exiting."
            # exit 1
        fi
    done
done

# Create the unified directory
mkdir -p "$TrainEventTracePath/$SplitDirName/continuous" 
mkdir -p "$TrainEventTracePath/$SplitDirName/discrete"
mkdir -p "$TestEventTracePath/$SplitDirName/continuous" 
mkdir -p "$TestEventTracePath/$SplitDirName/discrete"

train_exists=true
test_exists=true

# Loop through provided suffixes
for i in "$@"; do
    for prefix in "train" "test"; do
        dir="$SCRATCH_DIR/${prefix}_event_traces_${i}"

        if [[ ! -d "$dir" ]]; then
            echo "Directory $dir does not exist, skipping."
            # Track which prefix doesn't exist (assuming that the job IDs exist)
            # if [[ "$prefix" == "train" ]]; then
            #     train_exists=false
            # else
            #     test_exists=false
            # fi
            continue
        fi

        # Select the correct destination based on prefix
        if [[ "$prefix" == "train" ]]; then
            DestPath="$TrainEventTracePath"
        else
            DestPath="$TestEventTracePath"
        fi

        # Copy continuous vars
        if [[ -d "$dir/$SplitDirName/continuous" ]]; then
            cp -r "$dir/$SplitDirName/continuous/"* "$DestPath"/$SplitDirName/continuous/ 2>/dev/null
            echo "Cleaning up directory ${dir} for job ID ${i} ..."
            # chmod -R u+r "$dir"
            # chmod -R u+w "$dir"
            rm -rf "$dir"
            # source ./cleanup_scratch.sh "${prefix}_event_traces_${i}"

            # find "$dir" -xtype l -delete 2>/dev/null  # Remove dangling symlinks
            # chmod -R u+w "$dir" 2>/dev/null            # Fix permissions where possible
            # rm -rf "$dir"/* 2>/dev/null                # Remove everything else
            # rmdir "$dir" 2>/dev/null || true

            # find "$dir" -type l -delete               # Remove ALL symlinks (dangling or not)
            # find "$dir" -mindepth 1 -delete 2>/dev/null  # Remove everything
        fi

        # Copy discrete vars
        if [[ -d "$dir/$SplitDirName/discrete" ]]; then
            cp -r "$dir/$SplitDirName/discrete/"* "$DestPath"/$SplitDirName/discrete/ 2>/dev/null
        fi
    done
done

# Zip both merged folders
cd "$(dirname "$TrainEventTracePath")"

if [ "$train_exists" = true ]; then
    echo "Zipping train event traces ..."
    zip -r "$(basename "$TrainEventTracePath").zip" "$(basename "$TrainEventTracePath")"   
    echo "Train event trace paths: ${TrainEventTracePath}.zip"
else
    echo "No train event traces to zip."
fi

if [ "$test_exists" = true ]; then
    echo "Zipping test event traces ..."
    zip -r "$(basename "$TestEventTracePath").zip" "$(basename "$TestEventTracePath")" 
    echo "Test event trace paths: ${TestEventTracePath}.zip"
else
    echo "No test event traces to zip."
fi

rm -rf "$TrainEventTracePath"
rm -rf "$TestEventTracePath"

# for i in "$@"; do
#     dir="$SCRATCH_DIR/event_traces_${i}"
#    echo "Cleaning up directory ${dir} for suffix ${i} ..."
#    rm -rf "$dir"
# done

# Automata Learning
module load java/21.0.1
exec_path=/global/home/hpc5649/hpm_automata_learning/heat-pump-mining-v4/build/distributions/heat-pump-mining-0.0.0-93-de0c128-SNAPSHOT/bin/heat-pump-mining
# exec_path=/global/home/hpc5649/hpm_automata_learning/heat-pump-mining-v3/build/distributions/heat-pump-mining-0.0.0-68-06d845e-SNAPSHOT/bin/heat-pump-mining

if [ "$train_exists" = true ]; then
    echo "Starting convert-format for train..."
    $exec_path convert-format -i "$(basename "$TrainEventTracePath").zip" -o "$(basename "$TrainEventTracePath").json.zst"
    rm "$(basename "$TrainEventTracePath").zip"
else
    echo "No train event traces to convert to .json.zst."
fi

if [ "$test_exists" = true ]; then
    echo "Starting convert-format for test..."
    $exec_path convert-format -i "$(basename "$TestEventTracePath").zip" -o "$(basename "$TestEventTracePath").json.zst"
    rm "$(basename "$TestEventTracePath").zip"
else
    echo "No test event traces to convert to .json.zst."
fi
