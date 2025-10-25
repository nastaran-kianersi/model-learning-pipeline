#!/bin/bash
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -t 0-00:45:00
#SBATCH --qos=privileged
#SBATCH --partition=reserved
#SBATCH --wait

export PYTHONUNBUFFERED=TRUE

infile_arg="$1"
shift
echo $infile_arg
EventTracePath="${infile_arg#--EventTracePath=}"

learner_args=("$@")
echo "${learner_args[@]}"

echo "Learner job started at: $(date)"

# Automata Learning
module load java/21.0.1
hpm_executable=build/distributions/heat-pump-mining-0.0.0-14-ad1ef39-SNAPSHOT/

cd ../../hpm_automata_learning/heat-pump-mining/
exec_path=build/distributions/heat-pump-mining-0.0.0-14-ad1ef39-SNAPSHOT/bin/heat-pump-mining

echo "The actual script started running at: $(date)"
echo "Starting convert-format..."
$exec_path convert-format -i $EventTracePath -o "$TMPDIR/nibe.json.zst"
echo "Starting select-and-merge..."
$exec_path select-and-merge -i "$TMPDIR/nibe.json.zst" -o "$TMPDIR/filtered.json.zst" -e $learner_args -s defrost -a
echo "Starting learn..."
$exec_path learn -i "$TMPDIR/filtered.json.zst" -o "$TMPDIR/nibe.dot"
echo "Starting validate..."
$exec_path validate -a "$TMPDIR/nibe.dot" -i "$TMPDIR/filtered.json.zst"

# Don't copy the files yet, you need to use them in the next step (but ./outputs does not exist;
# it must copy to the parent process's ./outputs, must come to this script as an input argument)
# cp -r "$TMPDIR/CPD_Analysis_Outputs/" ./outputs/

echo "Learner job ended at: $(date)"