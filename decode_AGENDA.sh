#!/bin/bash

if [ "$#" -lt 4 ]; then
  echo "./decode_AGENDA.sh <model> <nodes-file> <graph-file> <output>"
  exit 2
fi

MODEL=$1
NODES_FILE=$2
GRAPH_FILE=$3
OUTPUT=$4

export OMP_NUM_THREADS=10

python -u graph2text/translate.py -model ${MODEL} \
-src ${NODES_FILE} \
-graph ${GRAPH_FILE} \
-output ${OUTPUT} \
-beam_size 5 \
-share_vocab \
-min_length 0 \
-max_length 430 \
-length_penalty wu \
-alpha 5 \
-verbose \
-batch_size 80 \
-gpu 0

cat ${OUTPUT} | sed -r 's/(@@ )|(@@ ?$)//g' > "${OUTPUT}_proc.txt"
mv "${OUTPUT}_proc.txt" ${OUTPUT}