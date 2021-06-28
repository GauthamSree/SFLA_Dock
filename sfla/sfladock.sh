set -e
lis=(Data/4dn4*)
for ja in "${lis[@]}"
do
	cry="$(cut -d'/' -f2 <<<"$ja").pdb"  #"${ja}.pdb" #"${ja}.pdb"
	sub="$(cut -d'_' -f1 <<<$(cut -d'/' -f2 <<<"$ja"))"  #${ja:$((${#ja}-7)):4}
	mkdir ${sub}
for h in {0..0}
do

	
	touch  "report_${sub}_${h}.txt"
	chain=($(python3 sfla.py --pdb ${ja} -n 100 | tr -d '[],'))
	echo ${chain[0]}

	ch=()
	for ci in $(seq 1 ${chain[0]})
	do
		eval a=${chain[ci]}
		ch+=(${a})
	done
	echo ${ch[@]}

	result=(native_${sub}/*)

	# Analysis of the results

	for i in "${result[@]}"
	do
		./DockQ.py ${ja}/${cry} $i  -native_chain1 ${ch[@]}  | tee -a ds.txt # -native_chain1 -native_chain1 A B
		result1=($(python read.py | tr -d '[],'))
		result1=("${result1[@]//\'/}")
		echo "$i  ${result1[@]}" >> "report_${sub}_${h}.txt"
		#printf "\n" >>"result_${sub}.txt"
		#arr += ${result[@]}

		rm ds.txt
	done

	tar cvfz "native_${sub}_${h}.tar.gz" "native_${sub}"
	mv native_${sub}_${h}.tar.gz report_${sub}_${h}.txt "${sub}"
	
	rm -r poses/*
	rm -r native_${sub}
	rm -r /tmp/tmp[a-d]*
	rm -r /tmp/tmp[e-j]*
	rm -r /tmp/tmp[k-r]*
	rm -r /tmp/tmp[s-z]*
	rm -r /tmp/tmp[0-9]*
	rm -r /tmp/tmp*
	#rm -r tmp*

done

touch "best_${sub}_${h}.txt"

li=(${sub}/report*)
for xi in "${li[@]}"
do
	mapfile -t array < <(python final.py -f ${xi})
	echo ${array[0]} >> "best_${sub}.txt"
done
mv best_${sub}.txt bestenergy.txt "${sub}"
rm -r *.pdb
rm -r Data/${sub}*
#rm -r /tmp/tmp*
#mv ${sub} "/home/sharon/IFPAresult50"
#mv ${ja} "/home/sharon/done_PPDBv5"
done


