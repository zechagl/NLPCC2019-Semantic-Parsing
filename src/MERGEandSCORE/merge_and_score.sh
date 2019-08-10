#!/bin/bash

set -x

export DATAHOME=./datahome
export EXEHOME=./codes

cd ${EXEHOME}

# transform timestep2 v1 to v2 : merge entity-label results into timestep2
python label_in_timestep2.py -mode dev \
       -input_path ${DATAHOME}/timestep2/15_classes_dev_timestep2_v1.json \
       -label_path ${DATAHOME}/label/dev_labeled_results_v2.json \
       -result_path ${DATAHOME}/timestep2/15_classes_dev_timestep2_v2.json

python label_in_timestep2.py -mode test \
       -input_path ${DATAHOME}/timestep2/15_classes_test_timestep2_v1.json \
       -label_path ${DATAHOME}/label/test_labeled_results_v2.json \
       -result_path ${DATAHOME}/timestep2/15_classes_test_timestep2_v2.json

# transform timestep2 v2 to v3 :
# align the number of entities/values predicted
# with that in type_pred_pattern in timestep2 data
python entity_in_timestep2.py -mode dev \
       -input_path ${DATAHOME}/timestep2/15_classes_dev_timestep2_v2.json \
       -label_path ${DATAHOME}/label/dev_labeled_results_v2.json \
       -result_path ${DATAHOME}/timestep2/15_classes_dev_timestep2_v3.json

python entity_in_timestep2.py -mode test \
       -input_path ${DATAHOME}/timestep2/15_classes_test_timestep2_v2.json \
       -label_path ${DATAHOME}/label/test_labeled_results_v2.json \
       -result_path ${DATAHOME}/timestep2/15_classes_test_timestep2_v3.json

# produce the merge files :
# merge the entity-relation scores into data
# merge the loss results into data
python merge_dev.py \
       -input_path ${DATAHOME}/timestep2/15_classes_dev_timestep2_v3.json \
       -pred_path ${DATAHOME}/pred/out.test.v2_ep69c_f1.dev_pred.results.json \
       -result_path ${DATAHOME}/merge/15_classes_dev_timestep2_merge.json \
       -loss_path ${DATAHOME}/loss/loss_results.json ${DATAHOME}/loss/results_16.json ${DATAHOME}/loss/results_22.json

python merge_test.py \
       -input_path ${DATAHOME}/timestep2/15_classes_test_timestep2_v3.json \
       -pred_path ${DATAHOME}/pred/out.test.v2_ep69c_f1.test_pred.results.json \
       -result_path ${DATAHOME}/merge/15_classes_test_timestep2_merge.json \
       -loss_path ${DATAHOME}/loss/test_results_12.json ${DATAHOME}/loss/test_results_16.json ${DATAHOME}/loss/test_results_22.json

# get the prediction logical form using union of scores
python score_merge_dev.py \
       -input_path ${DATAHOME}/merge/15_classes_dev_timestep2_merge.json \
       -result_path result.dev.txt \
       -error_path error.dev.txt \
       -todo1to3 0.33333333333 -todo1to3_aggregation 25 -todo1to3_singlerelation 0.125 \
       -todo1to3_superlative0 90 -todo1to3_multichoice 0.04 \
       -cover2todo13 0.0025 -loss2others 0.5 -loss2others_aggregation 0.001

python score_merge_test.py \
       -input_path ${DATAHOME}/merge/15_classes_test_timestep2_merge.json \
       -result_path result.test.txt \
       -todo1to3 0.33333333333 -todo1to3_aggregation 25 -todo1to3_singlerelation 0.125 \
       -todo1to3_superlative0 90 -todo1to3_multichoice 0.04 \
       -cover2todo13 0.0025 -loss2others 0.5 -loss2others_aggregation 0.001
