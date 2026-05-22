#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 0-00:15:00
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH --error=/dev/null
#SBATCH --output=/dev/null
##SBATCH --wait
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

# desiredvar_arg="$1"
# shift
# echo $desiredvar_arg
# desired_var="${desiredvar_arg#--desired_var=}"

uv_args=("$@")
echo "${uv_args[@]}"

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
propfile_name=$(basename "$propfile_addr")
propfile_local_addr="$TMPDIR/$propfile_name"
cp "$propfile_addr" "$propfile_local_addr"

# Open the propfile_local_addr file and modify its content
# mapfile -t lines < "$propfile_local_addr"
# : > "$propfile_local_addr"  # empty the file

# for line in "${lines[@]}"; do
#     if [[ $line != "$desired_var"* ]]; then
#         echo "#$line" >> "$propfile_local_addr"
#     else
#         echo "$line" >> "$propfile_local_addr"
#     fi
# done

# uv run event-trace-extraction-by-CPD "$TMPDIR"/$(basename "$infile_addr") "$propfile_local_addr" --base-dir "$TMPDIR/CPD_Analysis_Outputs" --data-timezone "Europe/Paris" --alg-params "${uv_args[@]}" | tee "$tmpfile"
# tmpfile=$(mktemp)
# OutputPath=$(tail -n 1 "$tmpfile")
# rm "$tmpfile"

# --split-time "08:00"
# OutputPath=$(uv run event-trace-extraction-by-CPD "$TMPDIR"/$(basename "$infile_addr") "$propfile_local_addr" --base-dir "$TMPDIR/CPD_Analysis_Outputs" --data-timezone "Europe/Paris" --alg-params "${uv_args[@]}" | tail -n 1)
uv run event-trace-extraction-by-CPD "$TMPDIR"/$(basename "$infile_addr") \
    "$propfile_local_addr" \
    --base-dir "$TMPDIR/CPD_Analysis_Outputs" \
    --data-timezone "Europe/Paris" \
    --split-time "08:00" \
    --alg-params "${uv_args[@]}"  \
    --enable-train-test-split --train-ratio=0.8
deactivate

# Read the file into an array (one element per line)
mapfile -t paths < /tmp/output_path.txt

# Process first line (always exists)
OutputPath="${paths[0]}"
echo "Train discretization output path: $OutputPath"

SCRATCH_DIR="/global/scratch/hpc5649"
EventTracePath="$SCRATCH_DIR/train_event_traces_$SLURM_JOB_ID"
echo "Train event trace path: $EventTracePath"
mkdir -p "$EventTracePath"
cp -r "$OutputPath"/* "$EventTracePath"

# Process second line if it exists
if [ ${#paths[@]} -eq 2 ]; then
    OutputPath2="${paths[1]}"
    echo "Test discretization output path: $OutputPath2"
    
    EventTracePath2="$SCRATCH_DIR/test_event_traces_$SLURM_JOB_ID"
    echo "Test event trace path: $EventTracePath2"
    mkdir -p "$EventTracePath2"
    cp -r "$OutputPath2"/* "$EventTracePath2"
fi

# mkdir -p ./outputs/event_traces_$SLURM_JOB_ID # prepare the target location
# cp -R "$OutputPath"/. ./outputs/event_traces_$SLURM_JOB_ID # copy only the contents
WAIT_TIME=$(($RANDOM % 3 + 1))
sleep $WAIT_TIME
echo "Particle job ended at: $(date)"
