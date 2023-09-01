##############02_catDFE.sh
#!/bin/bash
helpFunction()
{
   echo -e "02_catDFE.sh: estimate rho(ρ) of genomic category using sfs"
   echo "Usage:sh $0 -a data/ -b category/ -p 0.1 -m 5000000 -t 32 -o traesA"
   echo -e "\t-a directory with preDFE files"
   echo -e "\t-b directory with categorial compressed bedgraph files"
   echo -e "\t-p top percent of category as neutral"
   echo -e "\t-m max length of genomic category <bp>"
   echo -e "\t-t threads"
   echo -e "\t-o output prefix"
exit 1
}

while getopts "a:b:p:m:t:o:" opt
do
   case "$opt" in
      a ) data="$OPTARG" ;;
      b ) category="$OPTARG" ;;
      p ) percent="$OPTARG" ;;
      m ) maxCat="$OPTARG" ;;
      t ) cpu="$OPTARG" ;;
      o ) out="$OPTARG" ;;
      ? ) helpFunction ;;
   esac
done



if [ -z "$data" ] || [ -z "$category" ] || [ -z "$percent" ]|| [ -z "$maxCat" ]  || [ -z "$cpu" ] || [ -z "$out" ]
then
   echo "Error:missing parameters";
   helpFunction
fi



export LD_LIBRARY_PATH=/data1/home/lipeng/miniconda3/lib/:$LD_LIBRARY_PATH
data=$data
thread=$cpu
category=$category
out=$out
percent=$percent
maxCat=$maxCat

mkdir dfe
#merge preDFE files
cat $data/*_DATA.txt > $data/${out}-DATA.txt
#split category
zcat $category/*.gz|awk  -v out=${out} '{print $1"\t"$2"\t"$3 > "dfe/"out"-"$4"-cat.txt"}'
#filter category with extremely large size
>dfe/category_size.txt
for i in dfe/*-cat.txt ;do awk -F'\t' 'BEGIN{SUM=0}{ SUM+=$3-$2 }END{print SUM"\t"FILENAME}' $i >>dfe/category_size.txt;done
awk -v maxCat=$maxCat '{if($1>maxCat)print $2}' dfe/category_size.txt | xargs rm
#extract categorial preDFE files
>extract.sh
for i in dfe/*-cat.txt;do echo "bedtools intersect -a data/"$out"-DATA.txt -b $i |cut -f4- >${i::-8}-DATA.txt" >>extract.sh; done
rush -j $thread {} -i extract.sh
rm extract.sh
#select category with at least 100 polarized polymorphism
wc -l dfe/*-DATA.txt|awk '{if($1<100)print $2}' | xargs rm
#estimate usfs
for i in dfe/*-DATA.txt;do src/est-sfs/est-sfs src/est-sfs/config-rate6.txt $i src/est-sfs/seedfile.txt ${i::-8}sfs.txt ${i::-8}pvalues.txt;done
#extract neutral-like sites using category with defined bottom proportion of sfs-class-1 and sfs-class-2
>dfe/${out}-pct.txt
for i in dfe/*-sfs.txt;do awk -F "," '{sum=0;for(i=1;i<=NF;i++){sum+=$i};print FILENAME"\t"($1+$2)/sum}' $i |awk '{split($1,a,"-");print a[2]"\t"$2}' >>dfe/${out}-pct.txt;done
catNum=$(wc -l <dfe/${out}-pct.txt)
catNEU=$(echo 1|awk -v catNum=$catNum -v percent=$percent '{print int(percent*catNum)}')
sort -k2,2n dfe/${out}-pct.txt|head -n $catNEU|cut -f1 |awk '{print "dfe/traes-"$1"-DATA.txt"}' |datamash transpose -t " " |xargs cat >dfe/${out}-NEU_DATA.txt
#estimate neutral usfs
src/est-sfs/est-sfs src/est-sfs/config-rate6.txt dfe/${out}-NEU_DATA.txt src/est-sfs/seedfile.txt dfe/${out}-NEU_sfs.txt  dfe/${out}-NEU_pvalues.txt
#prepare combined sfs file for dfe-alpha
for i in dfe/*-sfs.txt;do cat <(echo 1) <(echo 20) $i  dfe/${out}-NEU_sfs.txt |sed "s/,/ /g" >${i::-7}2sfs.txt ;done
cat <(echo 1) <(echo 20) dfe/${out}-NEU_sfs.txt  dfe/${out}-NEU_sfs.txt |sed "s/,/ /g" >dfe/${out}-NEU_2sfs.txt
#estimate neutral-like dfe
sed "s/sfs.txt/dfe\/${out}-NEU_2sfs.txt/g" src/dfe-alpha/example-config-file-for-est_dfe-site_class-0.txt >src/dfe-alpha/est_dfe-site_class-0.txt
src/dfe-alpha/est_dfe -c src/dfe-alpha/est_dfe-site_class-0.txt
#estimate dfe of category
for i in dfe/*-2sfs.txt;do
j=${i:4}
sed "s/sfs.txt/dfe\/$j/g" src/dfe-alpha/example-config-file-for-est_dfe-site_class-1.txt >src/dfe-alpha/est_dfe-site_class-1.txt
src/dfe-alpha/est_dfe -c src/dfe-alpha/est_dfe-site_class-1.txt
src/dfe-alpha/prop_muts_in_s_ranges -c src/dfe-alpha/results_dir_sel/est_dfe.out -o dfe/${j::-4}-p.txt
done
#merge p value
mkdir $out
>$out/${out}-p.txt
for i in dfe/*-p.txt;do
rho=$(awk '{print $6+$9+$12}' $i)
echo -e "${i::-6}\t$rho"|awk '{split($1,a,"-");printf ("%s\t%.3f\n",a[2],$2)}' >>$out/${out}-p.txt
done
#pull back p score
for i in $category/*.gz;do
j=$(echo $i|awk '{split($1,a,"/");print a[length(a)]}')
csvtk join -f1 -H -t <(zcat $i |awk '{print $4"\t"$0}') <(cat $out/${out}-p.txt) |cut -f2-|bgzip >$out/${j::-7}-p.bed.gz
done
#clean
rm stats.out 
echo "The rho files are stored in dir:"$out"/"
echo "done."
