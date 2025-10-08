#!/bin/bash

echo "#!/bin/bash" > jobscript.sh
echo "#SBATCH -o ${1}job_%A.out" >> jobscript.sh
echo "#SBATCH -e ${1}job_%A.err" >> jobscript.sh

cat $2 >> jobscript.sh

# tmp_arg="$1"
# shift
# tmp_arg="$2"
# shift
# jobscript_args=("$@")
# echo "${jobscript_args[@]}"

#submit the job
# sbatch jobscript.sh "$@"
# output=$(sbatch jobscript.sh "$@")
# echo "$output"