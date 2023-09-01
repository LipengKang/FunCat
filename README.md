# FunCat
predicting Functional effects using Categorical multi-omics data

## Install
#create conda environment
conda create -n funCat

conda activate funCat

conda install -c bioconda bedops

conda install -c bioconda ucsc-mafsinregion

conda install -c bioconda ucsc-mafranges

conda install -c bioconda plink2

conda install -c bioconda phast

#install parallel tool

wget https://github.com/shenwei356/rush/releases/download/vx.x.x/rush_xxx.tar.gz

tar -zxvf rush_xxx.tar.gz

mv rush /directoryInEnvironment

#install DFE-alpha
#Install dfe-alpha and est-sfs following their documents

dfe-alpha:https://sourceforge.net/projects/dfe-alpha-k-e-w/

est-sfs:https://sourceforge.net/projects/est-usfs/

#download and uncompress data files

dfe-alpha data files:https://datashare.ed.ac.uk/handle/10283/2730

unzip DS_10283_2730.zip

tar xvzf data.tar.gz
 
 
#place compiled binary and data files to funCat src

mv dfe-alpha-release-2.16/prop_muts_in_s_ranges funcat_v1.0/src/dfe-alpha/prop_muts_in_s_ranges

mv dfe-alpha-release-2.16/est_dfe funcat_v1.0/src/dfe-alpha/est_dfe

mv est-sfs-release-2.03/est-sfs funcat_v1.0/src/est-sfs/est-sfs

mv data funcat_v1.0/src/dfe-alpha/

cp -r funcat_v1.0/src/dfe-alpha/data funcat_v1.0/src/dfe-alpha/data-three-epoch



## Examples

step0: reconstruct ancestral genome and prepare DFE files

bash 01_preDFE.sh -a chr1_test.vcf.gz -b chr1_test.maf.gz -m test_named.mod -r reference -n nodeName -o chr1

step1: estimate rho(negative selection) using categorial multi-oimcs data

bash 02_catDFE.sh -a data/ -b class/ -p 0.15 -m 5000000 -t 32 -o outprefix

step3: training 

python3 lincat.py -i ./data/ -t sl -c 31 -m ./sl.model -j 100

step3: predicting 

python3 lincat.py -i ./data/ -t sl -c 31 -m ./sl.model -j 100
