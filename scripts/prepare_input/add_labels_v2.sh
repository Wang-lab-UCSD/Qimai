#!/bin/bash
cd "$1"
# save original IFS value so we can restore it later
oIFS="$IFS"
IFS="_"
declare -a fields=($2'.fasta')
IFS="$oIFS"
unset oIFS

pro=${fields[1]}
echo $pro
awk '{print $4}' $2'.bed' > $2'_tmp0.txt'
$HOME'/DPI/dataset/Encode3/FastaToTbl' $2'.fasta' | paste -d'\t' - $2'_tmp0.txt' |sed "s/$/\t$pro/" > $2'_tmp1.txt'
$HOME'/DPI/dataset/Encode3/FastaToTbl' $2'_shuffled.fasta' | sed "s/$/\t0\t$pro/" > $2'_tmp2.txt'

cat $2'_tmp1.txt' $2'_tmp2.txt' | sort > $2'.csv'
# cat $2'_tmp1.txt' $2'_tmp2.txt' > $2'.csv'


rm $2'_tmp0.txt'
rm $2'_tmp1.txt'
rm $2'_tmp2.txt'
