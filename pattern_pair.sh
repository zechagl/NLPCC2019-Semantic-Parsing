TRAIN_FULL=data/patternpair/train_all.tsv
LINES=data/json/line_numbers.json
TRAIN_PREDICT=data/json/patternpair_train_predict.json
MODE=l
SELECT=data/json/patternpair_select.json
TRAIN_NEW=data/patternpair/train_7epoch.tsv
#time python -m src.patternpair.get_new_train_file --train_full=$TRAIN_FULL --lines=$LINES --train_predict=$TRAIN_PREDICT --mode=$MODE --select=$SELECT --train_new=$TRAIN_NEW
#
#MODE=n0
#time python -m src.patternpair.get_new_train_file --train_full=$TRAIN_FULL --lines=$LINES --train_predict=$TRAIN_PREDICT --mode=$MODE --select=$SELECT --train_new=$TRAIN_NEW
#
#MODE=t
#time python -m src.patternpair.get_new_train_file --train_full=$TRAIN_FULL --lines=$LINES --train_predict=$TRAIN_PREDICT --mode=$MODE --select=$SELECT --train_new=$TRAIN_NEW

export CUDA_VISIBLE_DEVICES=7
BERT_BASE_DIR=uncased_L-12_H-768_A-12
DATA_DIR=data/patternpair/
OUTPUT_DIR=output/patternpair/7epoch_udf
EPOCH_NUM=1.0
BATCH_SIZE=32
LEARNING_RATE=2e-5
MAX_LENGTH=128
DROPOUT_RATE=0.1
OUTPUT_FILE_NAME=dev.json
TRAIN_FILE_NAME=$TRAIN_NEW
TEST_FILE_NAME=dev.tsv
OLD_DIR=output/patternpair/6epoch_udf

time python -m src.patternpair.mspars_relation \
--task_name=mspars --do_train=false --do_eval=false --do_predict=true --data_dir=$DATA_DIR \
--vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$OLD_DIR/model.ckpt-35338 \
--train_file_name=$TRAIN_FILE_NAME --test_file_name=$TEST_FILE_NAME --output_file_name=$OUTPUT_FILE_NAME --dropout_rate=$DROPOUT_RATE --max_seq_length=$MAX_LENGTH --train_batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --num_train_epochs=$EPOCH_NUM --output_dir=$OUTPUT_DIR --predict_batch_size 32

cd src/patternpair

time python merge.py
time python comb.py


EXEHOME=src/MERGEandSCORE/code
DATAHOME=data
cd ${EXEHOME}

###===== produce the merge files : merge the entity-relation scores into data =====###
###                                merge the loss results into data                ###
python merge.py \
       -mode dev \
       -input_path ${DATAHOME}/json/timestep2_dev_v3.json \
       -pred_path ${DATAHOME}/json/pattern_pair_merge_5_dev.json \
       -result_path ${DATAHOME}/json/timestep3_dev.json \
       -loss_path ${DATAHOME}/output/pointer/dev_epoch20.json data/output/pointer/dev_epoch26.json data/output/pointer/dev_epoch29.json \
       -index_path ${DATAHOME}/json/dev_0_dict.json data/json/dev_1_dict.json \
       -qu2logical ${DATAHOME}/json/qu2logical.json

python merge.py \
       -mode test \
       -input_path ${DATAHOME}/json/timestep2_test_v3.json \
       -pred_path ${DATAHOME}/json/pattern_pair_merge_5_test.json \
       -result_path ${DATAHOME}/json/timestep3_test.json \
       -loss_path ${DATAHOME}/output/pointer/test_epoch20.json data/output/pointer/test_epoch26.json data/output/pointer/test_epoch29.json \
       -index_path ${DATAHOME}/json/test_0_dict.json data/json/test_1_dict.json \
       -qu2logical ${DATAHOME}/json/qu2logical.json
