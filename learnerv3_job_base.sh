#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0-00:15:00
#SBATCH --qos=privileged
#SBATCH --partition=reserved
##SBATCH --error=/dev/null
##SBATCH --wait

infile_arg="$1"
shift
echo $infile_arg
EventTraceFilename="${infile_arg#--EventTraceFilename=}"

train_test_split="true"
    
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

timing_eps="$1"
shift
timing_eps_decay="$1"
shift
freq_eps="$1"
shift
freq_eps_decay="$1"
shift
tail_length="$1"
shift

event_sel_args=("$@")
echo "${event_sel_args[@]}"

echo "Learner job started at: $(date)"

# Automata Learning
module load java/21.0.1
cd ../../hpm_automata_learning/heat-pump-mining-v3/
exec_path=build/distributions/heat-pump-mining-0.0.0-68-06d845e-SNAPSHOT/bin/heat-pump-mining

echo "The actual script started running at: $(date)"
if [[ "$train_test_split" = "false" ]]; then
    # Train only (no test file)
    echo "Starting convert-format..."
    $exec_path convert-format -i $TrainEventTracePath -o "$TMPDIR/nibe.json.zst"
    echo "Starting select-and-merge..."
    last_arg="${event_sel_args[@]: -1}"
    # echo "Last arg: $last_arg, learner args: ${event_sel_args[@]}"
    $exec_path select-and-merge -i "$TMPDIR/nibe.json.zst" -o "$TMPDIR/filtered.json.zst" -e "${event_sel_args[@]}" -s "$last_arg" --use-all-non-overlapping
    echo "Starting learn..."
    if [ "$tail_length" -eq 0 ]; then
        tail_length_opt=""
    else
        tail_length_opt="--tail-length=$tail_length"
    fi
    $exec_path learn -i "$TMPDIR/filtered.json.zst" -o "$TMPDIR/nibe.dot" --init=$init -d -u \
        --timing-epsilon="$timing_eps" --timing-epsilon-decay="$timing_eps_decay" --frequency-epsilon="$freq_eps" --frequency-epsilon-decay="$freq_eps_decay" $tail_length_opt
    
    echo "Starting validate-revision-score..."
    $exec_path compute-revision-score -a "$TMPDIR/nibe.dot" -i "$TMPDIR/filtered.json.zst" --init=$init --single-rooted #--parallel
else
    # Train and test
    echo "Starting convert-format for train..."
    $exec_path convert-format -i $TrainEventTracePath -o "$TMPDIR/train_nibe.json.zst"
    echo "Starting convert-format for test..."
    $exec_path convert-format -i $TestEventTracePath -o "$TMPDIR/test_nibe.json.zst"
    echo "Starting select-and-merge for train..."
    last_arg="${event_sel_args[@]: -1}"
    # echo "Last arg: $last_arg, learner args: ${event_sel_args[@]}"
    $exec_path select-and-merge -i "$TMPDIR/train_nibe.json.zst" -o "$TMPDIR/train_filtered.json.zst" -e "${event_sel_args[@]}" -s "$last_arg" --use-all-non-overlapping
    echo "Starting select-and-merge for test..."
    $exec_path select-and-merge -i "$TMPDIR/test_nibe.json.zst" -o "$TMPDIR/test_filtered.json.zst" -e "${event_sel_args[@]}" -s "$last_arg" --use-all-non-overlapping
    echo "Starting learn..."
    if [ "$tail_length" -eq 0 ]; then
        tail_length_opt=""
    else
        tail_length_opt="--tail-length=$tail_length"
    fi
    $exec_path learn -i "$TMPDIR/train_filtered.json.zst" -o "$TMPDIR/nibe.dot" --init=$init -d -u \
        --timing-epsilon="$timing_eps" --timing-epsilon-decay="$timing_eps_decay" --frequency-epsilon="$freq_eps" --frequency-epsilon-decay="$freq_eps_decay" $tail_length_opt

    echo "Starting validate-revision-score with train data..."
    $exec_path compute-revision-score -a "$TMPDIR/nibe.dot" -i "$TMPDIR/train_filtered.json.zst" --init=$init --single-rooted #--parallel
    echo "Starting validate-revision-score with test data..."
    $exec_path compute-revision-score -a "$TMPDIR/nibe.dot" -i "$TMPDIR/test_filtered.json.zst" --init=$init --single-rooted #--parallel
fi

echo "Rendering automaton..."
$exec_path render-automaton -i "$TMPDIR/nibe.dot" -o "$TMPDIR/neat.dot"
cp -r "$TMPDIR/neat.dot" "$SCRATCH_DIR"/"$outfile_addr"
echo "Learner job ended at: $(date)"