#set -e

# lis=(NewData12/*)
lis=(Data/4dn4*)
# lis=(NewData/1ahw*)
mkdir -p Result5

for DATAPTH in "${lis[@]}"; do
	EXTRACTED="$(cut -d '/' -f 2 <<< $DATAPTH)"
	FILE="$EXTRACTED.pdb" 
	PROTEIN="$(cut -d '_' -f 1 <<< $EXTRACTED)"
	mkdir -p ${PROTEIN}

	for h in {0..0}; do
		#Python SFLA
		#python3 sfla.py --pdb ${DATAPTH} -n 1
		echo "Starting ${PROTEIN}"
		VAL=($(python3 sfla.py --pdb ${DATAPTH} -n 100 | tr -d '[],'))

		CHAIN="$(cut -d ':' -f 1 <<< $(cut -d '_' -f 2 <<< $EXTRACTED))"
		CHAINARR=$(echo "$CHAIN" | grep -o .)
		RESULT=(native_${PROTEIN}/*)
		touch "report_${PROTEIN}_${h}.txt"

		#Analysis of the results
		for i in "${RESULT[@]}"; do
			./DockQ.py ${DATAPTH}/${FILE} $i -native_chain1 ${CHAINARR[@]} | tee -a ds.txt # -native_chain1 A B
			result1=($(python read.py | tr -d '[],'))
			result1=("${result1[@]//\'/}")
			echo "$i  ${result1[@]}" >> "report_${PROTEIN}_${h}.txt"
			cat ds.txt >> ${PROTEIN}/ds.txt
			rm ds.txt
		done
	done
	mv best_energy.txt "report_${PROTEIN}_${h}.txt" all_frog_energy.txt debug.log ${PROTEIN}/
	mv poses/ ${PROTEIN}/
	mv native_${PROTEIN}/ ${PROTEIN}/
	mv ${PROTEIN}/ Result5/

done

