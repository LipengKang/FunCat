# FunCat
predicting Functional effects using Categorical multi-omics data

## Install
* create conda environment
```sh
conda create -n funCat
conda activate funCat
conda install -c bioconda bedops
conda install -c bioconda ucsc-mafsinregion
conda install -c bioconda ucsc-mafranges
conda install -c bioconda plink2
conda install -c bioconda phast
```

* install parallel tool
```sh
wget https://github.com/shenwei356/rush/releases/download/vx.x.x/rush_xxx.tar.gz
tar -zxvf rush_xxx.tar.gz
mv rush /directoryInEnvironment
```

* install DFE-alpha

Install dfe-alpha and est-sfs following their documents

**dfe-alha++:** https://sourceforge.net/projects/dfe-alpha-k-e-w/

**est-sfs:** https://sourceforge.net/projects/est-usfs/

* install DFE-data files
 ```sh
wget https://datashare.ed.ac.uk/handle/10283/2730
unzip DS_10283_2730.zip
tar xvzf data.tar.gz
mkdir src
mv dfe-alpha-release-2.16/prop_muts_in_s_ranges funcat_v1.0/src/dfe-alpha/prop_muts_in_s_ranges
mv dfe-alpha-release-2.16/est_dfe funcat_v1.0/src/dfe-alpha/est_dfe
mv est-sfs-release-2.03/est-sfs funcat_v1.0/src/est-sfs/est-sfs
mv data funcat_v1.0/src/dfe-alpha/
cp -r funcat_v1.0/src/dfe-alpha/data funcat_v1.0/src/dfe-alpha/data-three-epoch
```

## Examples

1. reconstruct ancestral genome and prepare DFE files
```sh
bash 01_preDFE.sh -a chr1_test.vcf.gz -b chr1_test.maf.gz -m test_named.mod -r reference -n nodeName -o chr1
```
2. estimate rho(negative selection) using categorial multi-oimcs data
```sh
bash 02_catDFE.sh -a data/ -b class/ -p 0.15 -m 5000000 -t 32 -o outprefix
```
3. train
```python
python3 03_trainAndPredict.py -i data/ -t sl -c 31 -m ./sl.model -j 100
```
4. predict
```python
python3 03_trainAndPredict.py -e -i ./data/ -t sl -c 31 -m ./sl.model -o sl.output
```


