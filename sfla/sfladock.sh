set -e

lis=(Data/4dn4*)
for DATAPTH in "${lis[@]}"
do
	EXTRACTED="$(cut -d '/' -f 2 <<< $DATAPTH)"
	FILE="$EXTRACTED.pdb" 
	PROTEIN="$(cut -d '_' -f 1 <<< $EXTRACTED)"
	mkdir -p ${PROTEIN}

	for h in {0..0}; do
		touch "report_${PROTEIN}_${h}.txt"
		#Python SFLA
		#python3 sfla.py --pdb ${DATAPTH} -n 1
		#VAL=($(python3 sfla.py --pdb ${DATAPTH} -n 100 | tr -d '[],'))
		VAL=($(python3 sfla_dfire.py --pdb ${DATAPTH} -n 100 | tr -d '[],'))

		CHAIN="$(cut -d ':' -f 1 <<< $(cut -d '_' -f 2 <<< $EXTRACTED))"
		CHAINARR=$(echo "$CHAIN" | grep -o .)
		RESULT=(native_${PROTEIN}/*)

		#Analysis of the results
		for i in "${RESULT[@]}"; do
			./DockQ.py ${DATAPTH}/${FILE} $i -native_chain1 ${CHAINARR[@]}| tee -a ds.txt # -native_chain1 A B
			result1=($(python read.py | tr -d '[],'))
			result1=("${result1[@]//\'/}")
			echo "$i  ${result1[@]}" >> "report_${PROTEIN}_${h}.txt"
			cat ds.txt >> ${PROTEIN}/ds.txt
			rm ds.txt
		done
	done
	mv best_energy.txt "report_${PROTEIN}_${h}.txt" all_frog_energy.txt debug.log ${PROTEIN}/
	mv poses/ ${PROTEIN}/
	# touch "best_${PROTEIN}_${h}.txt"

	# li=(${PROTEIN}/report*)
	# for xi in "${li[@]}"; do
	# 	mapfile -t array < <(python final.py -f ${xi})
	# 	echo ${array[0]} >> "best_${PROTEIN}.txt"
	# done

done

