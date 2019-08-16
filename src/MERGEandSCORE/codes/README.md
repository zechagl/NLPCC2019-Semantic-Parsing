# How to run MERGEandSCORE


## Step1: ```multi_task_model.sh```

* transform timestep2 v1 to v2

    - ```label_in_timestep2.py```

    - merge entity-label results into timestep2

* transform timestep2 v2 to v3

    - ```entity_in_timestep2.py```
    
    - align the number of entities / values predicted with that in type_pred_pattern in timestep2-v2

## Step2: ```pattern_pair.sh```

* produce the merge files

    - ```merge.py```

    - merge the entity-predicate pair-scores into data

    - merge the pointer-loss scores into data: **you can choose which loss files to use here, however, at the present the number of loss files is fixed to 3**

## Step3: ```score.sh```

* get the predicted logical form using union of scores

    - ```score_merge.py```

    - there are currently four kinds of settings

        1. **WLIS_NEW** : baseline model

        2. **WLIS_NEW** + **point** : add point-loss as a score to consider for final prediction

        3. **WLIS_NEW** + **pep** : add entity-predicate pair-score to consider for final prediction

        4. **WLIS_NEW** + **point** + **pep** : add all scores for final prediction (including cover count)

## Step4: ```analysis```

* results of evaluation on development set are under the directory ```analysis```

* prediction results of test and dev sets are under the directory of ```data/output/results```

* error samples on dev set are under the directory of ```data/output/error```
