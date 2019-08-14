export CUDA_VISIBLE_DEVICES=1

BERT_BASE_DIR=uncased_L-12_H-768_A-12
DATA_DIR=data/pep
OUTPUT_DIR=output/pep/entity0
EPOCH_NUM=3.0
BATCH_SIZE=32
LEARNING_RATE=2e-5
MAX_LENGTH=128
DROPOUT_RATE=0.1
OUTPUT_FILE_NAME=dev_0.json
TRAIN_FILE_NAME=train_entity0.tsv
TEST_FILE_NAME=dev_entity0.tsv
PREDICT_BATCH_SIZE=32
python -m src.pep.mspars_pep --task_name=mspars --predict_batch_size=$PREDICT_BATCH_SIZE --do_train=false --do_eval=false --do_predict=true --data_dir=$DATA_DIR --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --train_file_name=$TRAIN_FILE_NAME --test_file_name=$TEST_FILE_NAME --output_file_name=$OUTPUT_FILE_NAME --dropout_rate=$DROPOUT_RATE --max_seq_length=$MAX_LENGTH --train_batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE --num_train_epochs=$EPOCH_NUM --output_dir=$OUTPUT_DIR

PREDICT=output/pep/entity0/dev_0.json
TSV=data/pep/dev_entity0.tsv
DICT=data/json/dev_0_dict.json
python -m src.pep.get_dict_file -p $PREDICT -t $TSV -d $DICT