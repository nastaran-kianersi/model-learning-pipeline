#SBATCH -c 2
#SBATCH --mem=3G
#SBATCH -t 0-00:08:00
#SBATCH --qos=privileged
#SBATCH --partition=reserved
##SBATCH --wait

infile_arg="$1"
shift
echo $infile_arg
EventTraceFilename="${infile_arg#--EventTraceFilename=}"

SCRATCH_DIR="/global/scratch/hpc5649"
EventTracePath="$SCRATCH_DIR/$EventTraceFilename"
echo "Full path to the event trace file: $EventTracePath"
# TODO: should I copy the input file from scratch memmory to $TMPDIR? 

# Derive a new filename from EventTracePath with a .dot extension,
# and set outfile_addr so the later cp uses that name.
orig_base="$(basename "$EventTracePath")"
base_no_ext="${orig_base%.*}"
outfile_addr="${base_no_ext}.dot"
echo "Prepared output filename: $outfile_addr"

learner_args=("$@")
echo "${learner_args[@]}"

echo "Learner job started at: $(date)"

# Automata Learning
module load java/21.0.1
cd ../../hpm_automata_learning/heat-pump-mining-v2/
exec_path=build/distributions/heat-pump-mining-0.0.0-41-6dbc376-SNAPSHOT/bin/heat-pump-mining

echo "The actual script started running at: $(date)"
echo "Starting convert-format..."
$exec_path convert-format -i $EventTracePath -o "$TMPDIR/nibe.json.zst"
echo "Starting select-and-merge..."
last_arg="${learner_args[-1]}"
$exec_path select-and-merge -i "$TMPDIR/nibe.json.zst" -o "$TMPDIR/filtered.json.zst" -e $learner_args -s $last_arg -a
echo "Starting learn..."
$exec_path learn -i "$TMPDIR/filtered.json.zst" -o "$TMPDIR/nibe.dot"
echo "Starting validate-revision-score..."
$exec_path validate-revision-score -a "$TMPDIR/nibe.dot" -i "$TMPDIR/filtered.json.zst"


cp -r "$TMPDIR/nibe.dot" "$SCRATCH_DIR"/"$outfile_addr"
echo "Learner job ended at: $(date)"