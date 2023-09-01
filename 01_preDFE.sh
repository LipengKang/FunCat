#########01_preDFE.sh
#!/bin/bash
helpFunction()
{
   echo -e "01_preDFE: prepare input for sfs related estimation"
   echo "Usage:sh $0 -a chr1D.vcf.gz -b 1D.maf.gz -m D_named.mod -r traesD -n traesD-aetau -o chr1D"
   echo -e "\t-a vcf <gz>"
   echo -e "\t-b maf <gz>"
   echo -e "\t-m mod file from phyloFit"
   echo -e "\t-r reference name in maf"
   echo -e "\t-n MRCA node name in mod"
   echo -e "\t-o chromosome name,also output prefix"
exit 1
}

while getopts "a:b:m:r:n:o:" opt
do
   case "$opt" in
      a ) vcf="$OPTARG" ;;
      b ) aln="$OPTARG" ;;
      m ) mod="$OPTARG" ;;
      r ) ref="$OPTARG" ;;
      n ) nod="$OPTARG" ;;
      o ) out="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done



if [ -z "$vcf" ] || [ -z "$aln" ] || [ -z "$mod" ] || [ -z "$ref" ] || [ -z "$nod" ] || [ -z "$out" ]
then
   echo "Error:missing parameters";
   helpFunction
fi


if [ ! -d pop ];then
mkdir pop
fi

if [ ! -d anc ];then
mkdir anc
fi

if [ ! -d data ];then
mkdir data
fi

chr=$out
ancNode=$nod

#get polymorphism info for dfe-alpha-datafile
plink2 --vcf $vcf --make-bed --out pop/$chr --allow-extra-chr --silent
cat pop/${chr}.bim  |awk '{print "chr"substr($1,1,1)"\t"$4-1"\t"$4}'|sortBed >anc/${chr}_snp_all.bed
plink2 --bfile pop/$chr --freq --allow-extra-chr --out pop/$chr --silent
#tail -n +2 pop/${chr}.afreq|awk '{A=0;C=0;G=0;T=0;split($2,a,"-");if($4=="G"){G=$5*100}else if($4=="A"){A=$5*100}else if($4=="C"){C=$5*100}else if($4=="T"){T=$5*100} print"chr"$1"\t"a[2]-1"\t"a[2]"\t"$3"\t"$4"\t"A","C","G","T}'|awk '{split($6,a,",");A=a[1];C=a[2];G=a[3];T=a[4];sum=a[1]+a[2]+a[3]+a[4];if($4=="A"){A=100-sum}else if($4=="C"){C=100-sum}else if($4=="G"){G=100-sum}else if($4=="T"){T=100-sum}printf "%s\t%s\t%s\t%.0f,%.0f,%.0f,%.0f\n", substr($1,1,4),$2,$3,A,C,G,T}'|sortBed  >pop/${chr}_pop.est
tail -n +2 pop/${chr}.afreq|awk '{A=0;C=0;G=0;T=0;split($2,a,"-");if($4=="G"){G=$5*20}else if($4=="A"){A=$5*20}else if($4=="C"){C=$5*20}else if($4=="T"){T=$5*20} print"chr"$1"\t"a[2]-1"\t"a[2]"\t"$3"\t"$4"\t"A","C","G","T}'|awk '{split($6,a,",");A=a[1];C=a[2];G=a[3];T=a[4];sum=a[1]+a[2]+a[3]+a[4];if($4=="A"){A=20-sum}else if($4=="C"){C=20-sum}else if($4=="G"){G=20-sum}else if($4=="T"){T=20-sum}printf "%s\t%s\t%s\t%.0f,%.0f,%.0f,%.0f\n", substr($1,1,4),$2,$3,A,C,G,T}'|sortBed  >pop/${chr}_pop.est

#get divergence info for dfe-alpha-datafile
mafsInRegion anc/${chr}_snp_all.bed anc/${chr}_tmp.maf $aln
grep -v 'null' anc/${chr}_tmp.maf|mafFilter -minRow=2 stdin > anc/${chr}_snp.maf
msa_view anc/${chr}_snp.maf --in-format MAF --out-format FASTA --gap-strip 1 --missing-as-indels >anc/${chr}_snp.fa
prequel -k -s $ancNode anc/${chr}_snp.fa $mod anc/${chr}
mafRanges anc/${chr}_snp.maf $ref anc/${chr}_snp_withAnc.bed
paste anc/${chr}_snp_withAnc.bed <(tail -n +2 anc/${chr}.${ancNode}.probs| awk '{max = $1;col = 1;if($1=="-")print $1;else {for (i = 2; i <= NF; i++){if ($i > max) {max = $i;col = i}};print col"\t"max}}'|awk '{if ($1 == 1) {print "1,0,0,0\t"$2"\tA";} else if ($1 == 2) {print "0,1,0,0\t"$2"\tC";} else if ($1 == 3) {print "0,0,1,0\t"$2"\tG";} else if ($1 == 4) {print "0,0,0,1\t"$2"\tT";} else {print "-\t-\t-"} }')  >anc/${chr}_anc.est
#prepare dfe-alpha-datafile
bedtools intersect -a  pop/${chr}_pop.est -b anc/${chr}_anc.est -wo |awk '{if(($9!="-")&&($9>0.8))print}'|cut -f1-4,8 |awk -v chr=$chr '{print chr"\t"$2"\t"$3"\t"$4"\t"$5}' >data/${chr}_DATA.txt
#prepare DAF
bedtools intersect -a <(tail -n +2 pop/${chr}.afreq|awk '{split($2,a,"-");print"chr"substr($1,1,1)"\t"a[2]-1"\t"a[2]"\t"$3"\t"$4"\t"$5}' ) -b anc/${chr}_anc.est -wo |awk  -v chr=$chr '{if($5==$12)print chr"\t"$2"\t"$3"\t"1-$6;else if($4==$12)print chr"\t"$2"\t"$3"\t"$6}'|awk '{printf ("%s\t%s\t%s\t%.3f\n",$1,$2,$3,$4)}' >pop/${chr}_DAF.bed
#prepare MAF
tail -n +2 pop/${chr}.afreq|awk '{split($2,a,"-");if($5>0.5)print "chr"$1"\t"a[2]-1"\t"a[2]"\t"1-$5;else print "chr"$1"\t"a[2]-1"\t"a[2]"\t"$5}'|awk '{printf ("%s\t%s\t%s\t%.3f\n",$1,$2,$3,$4)}' >pop/${chr}_MAF.bed
echo "The preDFE file is stored in dir:data/ "
echo "done."
