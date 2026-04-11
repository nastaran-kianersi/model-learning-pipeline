#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0-00:10:00
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
base_no_ext="${orig_base%.json.zst}" #base_no_ext="${orig_base%.*}"
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
# cd ../../hpm_automata_learning/heat-pump-mining-v3/
# exec_path=build/distributions/heat-pump-mining-0.0.0-68-06d845e-SNAPSHOT/bin/heat-pump-mining
cd ../../hpm_automata_learning/heat-pump-mining-v4/
exec_path=build/distributions/heat-pump-mining-0.0.0-93-de0c128-SNAPSHOT/bin/heat-pump-mining

echo "The actual script started running at: $(date)"
if [[ "$train_test_split" = "false" ]]; then
    # Train only (no test file)
    echo "Starting select-and-merge..."
    last_arg="${event_sel_args[@]: -1}"
    # echo "Last arg: $last_arg, learner args: ${event_sel_args[@]}"
    $exec_path select-and-merge -i $TrainEventTracePath -o "$TMPDIR/filtered.json.zst" -e "${event_sel_args[@]}" -s "$last_arg" --use-all-non-overlapping #--key-event-with-prefix="$last_arg" #--start-of-day-utc 08:00 #
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
    echo "Starting select-and-merge for train..."
    last_arg="${event_sel_args[@]: -1}"
    # echo "Last arg: $last_arg, learner args: ${event_sel_args[@]}"
    $exec_path select-and-merge -i $TrainEventTracePath -o "$TMPDIR/train_filtered.json.zst" -e "${event_sel_args[@]}" --start-of-day-utc 08:00 #--key-event-with-prefix="$last_arg" # -s "$last_arg"
    echo "Starting select-and-merge for test..."
    $exec_path select-and-merge -i $TestEventTracePath -o "$TMPDIR/test_filtered.json.zst" -e "${event_sel_args[@]}" --start-of-day-utc 08:00 #--key-event-with-prefix="$last_arg" # -s "$last_arg"
    echo "Starting learn..."
    if [ "$tail_length" -eq 0 ]; then
        tail_length_opt=""
    else
        tail_length_opt="--tail-length=$tail_length"
    fi
    $exec_path learn -i "$TMPDIR/train_filtered.json.zst" -o "$TMPDIR/nibe.dot" --init=$init --parallel -d -u \
        --timing-epsilon="$timing_eps" --timing-epsilon-decay="$timing_eps_decay" --frequency-epsilon="$freq_eps" --frequency-epsilon-decay="$freq_eps_decay" $tail_length_opt

    echo "Starting validate-revision-score with train data..."
    $exec_path compute-revision-score -a "$TMPDIR/nibe.dot" -i "$TMPDIR/train_filtered.json.zst" --init=$init --single-best --parallel
    echo "Starting validate-revision-score with test data..."
    $exec_path compute-revision-score -a "$TMPDIR/nibe.dot" -i "$TMPDIR/test_filtered.json.zst" --init=$init --single-best --parallel
fi

# echo "Starting compute-timing-dispersion..."
# $exec_path compute-timing-dispersion -a "$TMPDIR/nibe.dot"
# echo "Starting compute-states-per-output..."
# $exec_path compute-states-per-output -a "$TMPDIR/nibe.dot"

echo "Rendering automaton..."
# revision_score=$(grep -oP 'revision_score["\s:=]+\K[0-9.]+' <<< "$(history 1)" | tail -1)
# echo "Revision score: $revision_score"
# $exec_path render-automaton -i "$TMPDIR/nibe.dot" -o "$TMPDIR/neat.dot" --render-min=0.01
# cp -r "$TMPDIR/neat.dot" "$SCRATCH_DIR"/"$outfile_addr"

WAIT_TIME=$(($RANDOM % 3 + 1))
sleep $WAIT_TIME
echo "Learner job ended at: $(date)"

