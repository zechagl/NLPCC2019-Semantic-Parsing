#SOURCE_PATH=data/MSParS/MSParS.dev
#TARGET_PATH=data/json/dev.json
#LABEL2ID_PATH=data/json/label2id.json
#python -m src.multitask.data2json -s $SOURCE_PATH -t $TARGET_PATH -l $LABEL2ID_PATH
#
#SOURCE_PATH=data/MSParS/MSParS.train
#TARGET_PATH=data/json/train.json
#python -m src.multitask.data2json -s $SOURCE_PATH -t $TARGET_PATH
#
#SOURCE_TRAIN=data/json/train.json
#SOURCE_DEV=data/json/dev.json
#TARGET_TRAIN=data/multitask/train.tsv
#TARGET_TEST=data/multitask/test.tsv
#LABEL2ID=data/json/label2id.json
#RANDOM=36
#python -m src.multitask.get_multi_task_data --strain $SOURCE_TRAIN --sdev $SOURCE_DEV --train $TARGET_TRAIN --test $TARGET_TEST --label $LABEL2ID --random $RANDOM

export CUDA_VISIBLE_DEVICES=0
BERT_BASE_DIR=uncased_L-12_H-768_A-12
DATA_DIR=data/multitask
OLD_OUTPUT_DIR=output/multitask
NEW_OUTPUT_DIR=output/multitask
EPOCH_NUM=10.0
BATCH_SIZE=32
LEARNING_RATE=2e-5
MAX_LENGTH=90
CLASS_DROPOUT_RATE=0.1
ENTITY_DROPOUT_RATE=0.1
ENTITY_LABELING_WEIGHT=2.0
PREDICT_BATCH_SIZE=32
python -m src.multitask.mspars_multi_task --predict_batch_size=$PREDICT_BATCH_SIZE --task_name=mspars --do_train=false --do_predict=true --data_dir=$DATA_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=$MAX_LENGTH --entity_weight=$ENTITY_LABELING_WEIGHT --class_dropout=$CLASS_DROPOUT_RATE --entity_dropout=$ENTITY_DROPOUT_RATE --train_batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --num_train_epochs=$EPOCH_NUM --output_dir=$NEW_OUTPUT_DIR

RESULT=output/multitask/labeled_results.json
TOKEN=output/multitask/token_split.txt
ELABEL2ID=output/multitask/label2id_reference.json
SLABEL2ID=data/json/label2id.json
ANALYSIS=analysis/multi_task
python -m src.multitask.deal_with_multi_task -r $RESULT -t $TOKEN -e $ELABEL2ID -s $SLABEL2ID -a $ANALYSIS


EXEHOME=src/MERGEansSCORE/code
cd ${EXEHOME}

###===== transform timestep2 v1 to v2 : merge entity-label results into timestep2 =====###
python label_in_timestep2.py \
       -mode dev \
       -input_path data/json/timestep2/dev_timestep2_v1.json \
       -label_path data/output/multitask/dev_labeled_results.json \
       -result_path data/json/timestep2/dev_timestep2_v2.json

python label_in_timestep2.py \
       -mode test \
       -input_path data/json/timestep2/test_timestep2_v1.json \
       -label_path data/output/multitask/test_labeled_results.json \
       -result_path  data/json/timestep2/test_timestep2_v2.json

###===== transform timestep2 v2 to v3 : align the number of entities/values predicted =====###
###                                     with that in type_pred_pattern in timestep2 data   ###
python entity_in_timestep2.py \
       -mode dev \
       -input_path data/json/timestep2/dev_timestep2_v2.json \
       -label_path data/output/multitask/dev_labeled_results.json \
       -result_path data/json/timestep2/dev_timestep2_v3.json

python entity_in_timestep2.py \
       -mode test \
       -input_path data/json/timestep2/test_timestep2_v2.json \
       -label_path data/output/multitask/test_labeled_results.json \
       -result_path data/json/timestep2/test_timestep2_v3.json
