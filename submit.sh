#!/bin/bash

echo "#!/bin/bash" > $3
# or %j
echo "#SBATCH -o ${1}job_%A.out" >> $3
echo "#SBATCH -e ${1}job_%A.err" >> $3

# these don't output anything and idk why
#echo ${2}
#echo ${3}

# cat $2 >> jobscript.sh
cat $2 >> $3

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