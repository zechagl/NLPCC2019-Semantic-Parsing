##===== get the prediction logical form using union of scores =====##

DATAHOME=data
EXEHOME=src/MERGEandSCORE/code
EVALHOME=analysis

cd ${EXEHOME}

### test several mechanisms on development set ###
# 1. baseline
python score_merge.py \
       -mode dev \
       -input_path ${DATAHOME}/json/timestep3_dev.json \
       -result_path ${DATAHOME}/output/result/result.dev.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 1.5 \  # all of these values of hyper-parameters above are default
       -error_path ${DATAHOME}/output/error/error.dev.txt \
       -eval_path ${EVALHOME}/baseline \
       -remove_pointer -remove_pair
# 2. baseline + pointer
python score_merge.py \
       -mode dev \
       -input_path ${DATAHOME}/json/timestep3_dev.json \
       -result_path ${DATAHOME}/output/result/result.dev.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 15 \  # all of these values of hyper-parameters above are default
       -error_path ${DATAHOME}/output/error/error.dev.txt \
       -eval_path ${EVALHOME}/pointer \
       -remove_pair
# 3. baseline + entity-predicate-pair
python score_merge.py \
       -mode dev \
       -input_path ${DATAHOME}/json/timestep3_dev.json \
       -result_path ${DATAHOME}/output/result/result.dev.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 1.5 \  # all of these values of hyper-parameters above are default
       -error_path ${DATAHOME}/output/error/error.dev.txt \
       -eval_path ${EVALHOME}/pep \
       -remove_pointer
# 4. baseline + pointer + entity-predicate-pair
python score_merge.py \
       -mode dev \
       -input_path ${DATAHOME}/json/timestep3_dev.json \
       -result_path ${DATAHOME}/output/result/result.dev.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 1.5 \  # all of these values of hyper-parameters above are default
       -error_path ${DATAHOME}/output/error/error.dev.txt \
       -eval_path ${EVALHOME}/bsln_pointer_pep

### predict on test set ###
python score_merge.py \
       -mode test \
       -input_path ${DATAHOME}/json/timestep3_test.json \
       -result_path ${DATAHOME}/output/result/result.test.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 1.5 
