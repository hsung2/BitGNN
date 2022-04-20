#!/bin/bash

# dataset filenames
filename=(
email-Eu-core
email-Eu-core-temporal
CollegeMsg
soc-sign-bitcoin-alpha
ca-GrQc
soc-sign-bitcoin-otc
p2p-Gnutella08
as-735
p2p-Gnutella09
wiki-Vote
p2p-Gnutella06
p2p-Gnutella05
ca-HepTh
p2p-Gnutella04
wiki-RfA
Oregon-1
Oregon-2
ca-HepPh
ca-AstroPh
p2p-Gnutella25
ca-CondMat
sx-mathoverflow
p2p-Gnutella24
cit-HepTh
as-caida
cit-HepPh
p2p-Gnutella30
email-Enron
loc-Brightkite
p2p-Gnutella31
soc-Epinions1
soc-sign-Slashdot081106
soc-Slashdot0811
soc-sign-Slashdot090216
soc-sign-Slashdot090221
soc-Slashdot0902
soc-sign-epinions
sx-askubuntu
sx-superuser
loc-Gowalla
amazon0302
email-EuAll
web-Stanford
com-DBLP
web-NotreDame
com-Amazon
amazon0312
amazon0601
amazon0505
higgs-twitter
web-BerkStan
web-Google
roadNet-PA
com-Youtube
wiki-talk-temporal
roadNet-TX
soc-Pokec
as-Skitter
wiki-topcats
roadNet-CA
wiki-Talk
sx-stackoverflow
com-Orkut
cit-Patents
com-LiveJournal
soc-LiveJournal1
)
# twitter7
# com-Friendster
# )

TARGET_DIR="./eval_data"

if [ -d "$TARGET_DIR" ]; then
   echo "'$TARGET_DIR' found and now copying files, please wait ..."
else
   echo "Warning: '$TARGET_DIR' NOT found. Create it"
   mkdir $TARGET_DIR
fi

OUTPUT=output_$1.csv
rm -rf $OUTPUT

HIDDEN=(64 128 256 512)
OUTUNIT=(2 4 8 16)

#### b2sr test
if [ $1 = "b2sr_test" ]; then
  echo "-----------------------------------"
  make b2srtest
  for (( i=0; i<${#filename[@]}; i++ ));
  do
    echo "SNAP/${filename[$i]}"
    wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/${filename[$i]}.tar.gz
    tar zxvf ${filename[$i]}.tar.gz
    mv ${filename[$i]}/${filename[$i]}.mtx ${TARGET_DIR}
    rm -rf ${filename[$i]}.tar.gz
    rm -rf ${filename[$i]}
    printf "${filename[$i]} " >> $OUTPUT
    echo "./b2srtest ${TARGET_DIR}/${filename[$i]}.mtx"
    ./b2srtest ${TARGET_DIR}/${filename[$i]}.mtx >> $OUTPUT
    echo >> $OUTPUT
    rm -rf ${TARGET_DIR}/${filename[$i]}.mtx
  done
else
  echo "try again"
fi

#### baseline
if [ $1 = "baseline" ]; then
  echo "-----------------------------------"
  make $1
  for (( i=0; i<${#filename[@]}; i++ ));
  do
    echo "SNAP/${filename[$i]}"
    wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/${filename[$i]}.tar.gz
    tar zxvf ${filename[$i]}.tar.gz
    mv ${filename[$i]}/${filename[$i]}.mtx ${TARGET_DIR}
    rm -rf ${filename[$i]}.tar.gz
    rm -rf ${filename[$i]}
    printf "${filename[$i]}," >> $OUTPUT
    for (( j=0; j<${#HIDDEN[@]}; j++ ));
    do
        echo "spmm/baseline ${TARGET_DIR}/${filename[$i]}.mtx ${HIDDEN[$j]}"
        spmm/baseline ${TARGET_DIR}/${filename[$i]}.mtx ${HIDDEN[$j]} >> $OUTPUT
    done
    echo >> $OUTPUT
    rm -rf ${TARGET_DIR}/${filename[$i]}.mtx
  done
else
  echo "try again"
fi

#### bin_bin
if [ $1 = "bin_bin" ]; then
  echo "-----------------------------------"
  make spmmbb
  for (( i=0; i<${#filename[@]}; i++ ));
  do
    echo "SNAP/${filename[$i]}"
    wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/${filename[$i]}.tar.gz
    tar zxvf ${filename[$i]}.tar.gz
    mv ${filename[$i]}/${filename[$i]}.mtx ${TARGET_DIR}
    rm -rf ${filename[$i]}.tar.gz
    rm -rf ${filename[$i]}
    printf "${filename[$i]}," >> $OUTPUT
    for (( j=0; j<${#HIDDEN[@]}; j++ ));
    do
        echo "spmm/spmmbb${OUTUNIT[$j]} ${TARGET_DIR}/${filename[$i]}.mtx ${HIDDEN[$j]}"
        spmm/spmmbb${OUTUNIT[$j]} ${TARGET_DIR}/${filename[$i]}.mtx ${HIDDEN[$j]} >> $OUTPUT
    done
    echo >> $OUTPUT
    rm -rf ${TARGET_DIR}/${filename[$i]}.mtx
  done
else
  echo "try again"
fi

#### bin_full
if [ $1 = "full_bin" ]; then
  echo "-----------------------------------"
  make spmmfb
  for (( i=0; i<${#filename[@]}; i++ ));
  do
    echo "SNAP/${filename[$i]}"
    wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/${filename[$i]}.tar.gz
    tar zxvf ${filename[$i]}.tar.gz
    mv ${filename[$i]}/${filename[$i]}.mtx ${TARGET_DIR}
    rm -rf ${filename[$i]}.tar.gz
    rm -rf ${filename[$i]}
    printf "${filename[$i]}," >> $OUTPUT
    for (( j=0; j<${#HIDDEN[@]}; j++ ));
    do
        echo "spmm/spmmfb${OUTUNIT[$j]} ${TARGET_DIR}/${filename[$i]}.mtx ${HIDDEN[$j]}"
        spmm/spmmfb${OUTUNIT[$j]} ${TARGET_DIR}/${filename[$i]}.mtx ${HIDDEN[$j]} >> $OUTPUT
    done
    echo >> $OUTPUT
    rm -rf ${TARGET_DIR}/${filename[$i]}.mtx
  done
else
  echo "try again"
fi

# #### full_full
HIDDEN=(32 64 128 256)
OUTUNIT=(1 2 4 8)

if [ $1 = "full_full" ]; then
  echo "-----------------------------------"
  make spmmff
  for (( i=0; i<${#filename[@]}; i++ ));
  do
    echo "SNAP/${filename[$i]}"
    wget https://suitesparse-collection-website.herokuapp.com/MM/SNAP/${filename[$i]}.tar.gz
    tar zxvf ${filename[$i]}.tar.gz
    mv ${filename[$i]}/${filename[$i]}.mtx ${TARGET_DIR}
    rm -rf ${filename[$i]}.tar.gz
    rm -rf ${filename[$i]}
    printf "${filename[$i]}," >> $OUTPUT
    for (( j=0; j<${#HIDDEN[@]}; j++ ));
    do
        echo "spmm/spmmff${OUTUNIT[$j]} ${TARGET_DIR}/${filename[$i]}.mtx ${HIDDEN[$j]}"
        spmm/spmmff${OUTUNIT[$j]} ${TARGET_DIR}/${filename[$i]}.mtx ${HIDDEN[$j]} >> $OUTPUT
    done
    echo >> $OUTPUT
    rm -rf ${TARGET_DIR}/${filename[$i]}.mtx
  done
else
  echo "try again"
fi