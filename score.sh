DATAHOME=data
EXEHOME=src/MERGEandSCORE/code

cd ${EXEHOME}

# get the prediction logical form using union of scores
python score_merge.py \
       -mode dev \
       -input_path ${DATAHOME}/json/timestep3_dev.json \
       -result_path ${DATAHOME}/output/result/result.dev.txt \
       -error_path ${DATAHOME}/output/error/error.dev.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 -loss2others 1.5 

python score_merge.py \
       -mode test \
       -input_path ${DATAHOME}/json/timestep3_dev.json \
       -result_path ${DATAHOME}/output/result/result.test.txt \
       -todo1to3 0.4 -todo1to3_aggregation 0.5 -todo1to3_singlerelation 2 \
       -todo1to3_superlative0 15 -todo1to3_multichoice 0.25 \
       -cover2todo13 0.01 -loss2others 1.5 
