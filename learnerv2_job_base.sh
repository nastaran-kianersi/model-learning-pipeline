#SBATCH -c 2
#SBATCH --mem=3G
#SBATCH -t 0-00:15:00
#SBATCH --qos=privileged
#SBATCH --partition=reserved
##SBATCH --error=/dev/null
##SBATCH --wait

infile_arg="$1"
shift
echo $infile_arg
EventTraceFilename="${infile_arg#--EventTraceFilename=}"

train_test_split=true
    
SCRATCH_DIR="/global/scratch/hpc5649"
EventTracePath="$SCRATCH_DIR/$EventTraceFilename"
echo "Full path to the event trace file: $EventTracePath"

# Derive a new filename from EventTracePath with a .dot extension,
# and set outfile_addr so the later cp uses that name.
orig_base="$(basename "$EventTracePath")"
base_no_ext="${orig_base%.*}"
outfile_addr="${base_no_ext}_${SLURM_JOB_ID}.dot"
echo "Prepared output filename: $outfile_addr"

TrainEventTracePath="$SCRATCH_DIR/train_$EventTraceFilename"
TestEventTracePath="$SCRATCH_DIR/test_$EventTraceFilename"

learner_args=("$@")
echo "${learner_args[@]}"

echo "Learner job started at: $(date)"

# Automata Learning
module load java/21.0.1
cd ../../hpm_automata_learning/heat-pump-mining-v2/
exec_path=build/distributions/heat-pump-mining-0.0.0-41-6dbc376-SNAPSHOT/bin/heat-pump-mining

echo "The actual script started running at: $(date)"
if [ "$train_test_split" = false ]; then
    # Train only (no test file)
    echo "Starting convert-format..."
    $exec_path convert-format -i $TrainEventTracePath -o "$TMPDIR/nibe.json.zst"
    echo "Starting select-and-merge..."
    last_arg="${learner_args[@]: -1}"
    # echo "Last arg: $last_arg, learner args: ${learner_args[@]}"
    $exec_path select-and-merge -i "$TMPDIR/nibe.json.zst" -o "$TMPDIR/filtered.json.zst" -e "${learner_args[@]}" -s "$last_arg" -a
    echo "Starting learn..."
    $exec_path learn -i "$TMPDIR/filtered.json.zst" -o "$TMPDIR/nibe.dot"
    echo "Starting validate-revision-score..."
    $exec_path validate-revision-score -a "$TMPDIR/nibe.dot" -i "$TMPDIR/filtered.json.zst"
else
    # Train and test
    echo "Starting convert-format for train..."
    $exec_path convert-format -i $TrainEventTracePath -o "$TMPDIR/train_nibe.json.zst"
    echo "Starting convert-format for test..."
    $exec_path convert-format -i $TestEventTracePath -o "$TMPDIR/test_nibe.json.zst"
    echo "Starting select-and-merge for train..."
    last_arg="${learner_args[@]: -1}"
    # echo "Last arg: $last_arg, learner args: ${learner_args[@]}"
    $exec_path select-and-merge -i "$TMPDIR/train_nibe.json.zst" -o "$TMPDIR/train_filtered.json.zst" -e "${learner_args[@]}" -s "$last_arg" -a
    echo "Starting select-and-merge for test..."
    $exec_path select-and-merge -i "$TMPDIR/test_nibe.json.zst" -o "$TMPDIR/test_filtered.json.zst" -e "${learner_args[@]}" -s "$last_arg" -a
    echo "Starting learn..."
    $exec_path learn -i "$TMPDIR/train_filtered.json.zst" -o "$TMPDIR/nibe.dot"

    echo "Starting validate-revision-score with train data..."
    $exec_path validate-revision-score -a "$TMPDIR/nibe.dot" -i "$TMPDIR/train_filtered.json.zst"
    echo "Starting validate-revision-score with test data..."
    $exec_path validate-revision-score -a "$TMPDIR/nibe.dot" -i "$TMPDIR/test_filtered.json.zst"
fi

#cp -r "$TMPDIR/nibe.dot" "$SCRATCH_DIR"/"$outfile_addr"
echo "Learner job ended at: $(date)"