##===== get the prediction logical form using union of scores =====##

### test several mechanisms on development set ###
# 1. baseline
python -m src.score.score_merge \
       -mode dev \
       -input_path data/json/timestep3_dev.json \
       -result_path output/result/result.dev.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 1.5 \
       -error_path output/error/error.dev.txt \
       -eval_path analysis/baseline \
       -remove_pointer -remove_pair
# 2. baseline + pointer
python -m src.score.score_merge \
       -mode dev \
       -input_path data/json/timestep3_dev.json \
       -result_path output/result/result.dev.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 15 \
       -error_path output/error/error.dev.txt \
       -eval_path analysis/pointer \
       -remove_pair
# 3. baseline + entity-predicate-pair
python -m src.score.score_merge \
       -mode dev \
       -input_path data/json/timestep3_dev.json \
       -result_path output/result/result.dev.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 1.5 \
       -error_path output/error/error.dev.txt \
       -eval_path analysis/pep \
       -remove_pointer
# 4. baseline + pointer + entity-predicate-pair
python -m src.score.score_merge \
       -mode dev \
       -input_path data/json/timestep3_dev.json \
       -result_path output/result/result.dev.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 1.5 \
       -error_path output/error/error.dev.txt \
       -eval_path analysis/bsln_pointer_pep

### predict on test set ###
python -m src.score.score_merge \
       -mode test \
       -input_path data/json/timestep3_test.json \
       -result_path output/result/result.test.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 \
       -loss2others 1.5
