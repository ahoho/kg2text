#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "./preprocess_WEBNLG.sh <dataset_folder> <tokenizer_path_or_name>"
  exit 2
fi

processed_data_folder='../data/webnlg/kg2text/span_mask-checkpoint_10000'
mkdir -p ${processed_data_folder}

python preprocess/generate_input_webnlg.py ${1} ${processed_data_folder} --tokenizer_path_or_name ${2}

python graph2text/preprocess.py -train_src ${processed_data_folder}/train-nodes.txt \
                       -train_graph ${processed_data_folder}/train-graph.txt \
                       -train_logit_db ../outputs/webnlg/cmlm-logits/cmlm-span_mask-checkpoint_10000-mlm_prob_0.1 \
                       -train_tgt ${processed_data_folder}/train-surfaces-bpe.txt \
                       -valid_src ${processed_data_folder}/dev-nodes.txt  \
                       -valid_graph ${processed_data_folder}/dev-graph.txt  \
                       -valid_tgt ${processed_data_folder}/dev-surfaces-bpe.txt \
                       -save_data ${processed_data_folder}/webnlg \
                       -tgt_vocab ${processed_data_folder}/vocab.txt \
                       -src_vocab ${processed_data_folder}/vocab.txt \
                       -tokenizer_path_or_name ${2} \
                       -src_seq_length 10000 \
                       -tgt_seq_length 10000 \
                       -dynamic_dict \
                       -share_vocab \
                       -overwrite



