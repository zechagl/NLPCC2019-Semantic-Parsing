#### v1
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* 其他的，分解 + 合并均考虑
    ```
        overall 77.63333333333333

        aggregation 64.90066225165563
        comparative 97.82608695652173
        cvt 76.73667205169629
        multi-choice 39.55223880597015
        multi-constraint 83.95904436860067
        multi-hop 84.87179487179488
        multi-turn-answer 93.39622641509435
        multi-turn-entity 67.36938588450963
        multi-turn-predicate 87.0
        single-relation 83.63294875234773
        superlative0 63.75661375661375
        superlative1 96.55172413793103
        superlative2 100.0
        superlative3 100.0
        yesno 86.66666666666667
    ```

#### v2
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* aggregation
    TODO1 : TODO3 = 25 : 1
* single-relation
    TODO1 : TODO3 = 1 : 25
* others
    TODO1 : TODO3 = 3 : 10
    ```
        overall 80.06

    *   aggregation 66.67
        comparative 97.83
        cvt 76.74
    *   multi-choice 38.81
        multi-constraint 84.64
        multi-hop 87.18
        multi-turn-answer 94.34
        multi-turn-entity 74.52
        multi-turn-predicate 89.0
    *   single-relation 86.58
    *   superlative 68.49
        superlative0 63.23
        superlative1 95.69
        superlative2 100.0
        superlative3 100.0
        yesno 85.67
    ```

#### v3
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* 添加覆盖词数score3
    ( TODO1 & TODO3 ) : score3 = 200 : 1
* aggregation
    TODO1 : TODO3 = 25 : 1
* single-relation
    TODO1 : TODO3 = 1 : 25
* others
    TODO1 : TODO3 = 3 : 10
    ```
        overall 80.4

        aggregation 66.78
        comparative 97.83
        cvt 77.06
        multi-choice 38.81
        multi-constraint 84.3
        multi-hop 87.44
        multi-turn-answer 94.34
        multi-turn-entity 75.25
        multi-turn-predicate 90.0
        single-relation 87.07
        superlative 68.49
        superlative0 63.23
        superlative1 95.69
        superlative2 100.0
        superlative3 100.0
        yesno 85.67
    ```

#### v4
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* 添加覆盖词数score3
    ( TODO1 & TODO3 ) : score3 = 200 : 1
* aggregation
    TODO1 : TODO3 = 25 : 1
* single-relation
    TODO1 : TODO3 = 1 : 25
* superlative0
    TODO1 : TODO3 = 28 : 1
* others
    TODO1 : TODO3 = 3 : 10
    ```
        overall 80.7

        aggregation 66.78
        comparative 97.83
        cvt 77.06
        multi-choice 38.81
        multi-constraint 84.3
        multi-hop 87.44
        multi-turn-answer 94.34
        multi-turn-entity 75.25
        multi-turn-predicate 90.0
        single-relation 87.07
        superlative 71.49
        superlative0 66.8
        superlative1 95.69
        superlative2 100.0
        superlative3 100.0
        yesno 85.67
    ```

#### v5
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* 添加覆盖词数score3
    ( TODO1 & TODO3 ) : score3 = 200 : 1
* aggregation
    TODO1 : TODO3 = 25 : 1
* single-relation
    TODO1 : TODO3 = 1 : 25
* superlative0
    TODO1 : TODO3 = 28 : 1
* multi-choice
    TODO1 : TODO3 = 1 : 50
* others
    TODO1 : TODO3 = 1 : 3
    ```
        overall 80.79

        aggregation 66.78
        comparative 97.83
        cvt 76.9
        multi-choice 44.03
        multi-constraint 84.98
        multi-hop 87.44
        multi-turn-answer 94.34
        multi-turn-entity 75.25
        multi-turn-predicate 90.0
        single-relation 87.07
        superlative 71.49
        superlative0 66.8
        superlative1 95.69
        superlative2 100.0
        superlative3 100.0
        yesno 85.67
    ```

#### v6
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* 添加覆盖词数score3
    * ( TODO1 & TODO3 ) : score3 = 200 : 1
* aggregation
    * TODO1 : TODO3 = 25 : 1
* single-relation
    * TODO1 : TODO3 = 1 : 25
* superlative0
    * TODO1 : TODO3 = 25 : 1
* multi-choice
    * TODO1 : TODO3 = 1 : 25
* others
    * TODO1 : TODO3 = 1 : 3

```
    overall 80.77

    aggregation 66.78
    comparative 97.83
    cvt 76.9
    multi-choice 44.03
    multi-constraint 84.98
    multi-hop 87.44
    multi-turn-answer 94.34
    multi-turn-entity 75.25
    multi-turn-predicate 90.0
    single-relation 87.07
    superlative 71.27
    superlative0 66.53
    superlative1 95.69
    superlative2 100.0
    superlative3 100.0
    yesno 85.67
```

#### v7
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* 添加覆盖词数score3
    * ( TODO1 & TODO3 ) : score3 = 400 : 1
* aggregation
    * TODO1 : TODO3 = 25 : 1
* single-relation
    * TODO1 : TODO3 = 1 : 8
* superlative0
    * TODO1 : TODO3 = 90 : 1
* multi-choice
    * TODO1 : TODO3 = 1 : 25
* others
    * TODO1 : TODO3 = 1 : 3

```
        overall 81.24

        aggregation 66.78
        comparative 100.0
        cvt 76.9
        multi-choice 44.03
        multi-constraint 84.64
        multi-hop 86.28
        multi-turn-answer 94.34
        multi-turn-entity 76.54
        multi-turn-predicate 88.0
        single-relation 88.03
        superlative 71.05
        superlative0 66.4
        superlative1 94.83
        superlative2 100.0
        superlative3 100.0
        yesno 87.67
```

#### v8
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* 添加 loss_results - score4 = (1 - gold_loss) * (1 - pred_loss)
    * ((TODO1 & TODO3) & score3) : score4 = 2 : 1
* 添加覆盖词数score3
    * ( TODO1 & TODO3 ) : score3 = 400 : 1
* aggregation
    * TODO1 : TODO3 = 25 : 1
* single-relation
    * TODO1 : TODO3 = 1 : 8
* superlative0
    * TODO1 : TODO3 = 90 : 1
* multi-choice
    * TODO1 : TODO3 = 1 : 25
* others
    * TODO1 : TODO3 = 1 : 3

```
        overall 82.26

        aggregation 64.57
        comparative 100.0
        cvt 77.87
        multi-choice 48.51
        multi-constraint 87.03
        multi-hop 91.54
        multi-turn-answer 94.34
        multi-turn-entity 74.79
        multi-turn-predicate 88.0
        single-relation 88.6
        superlative 76.06
        superlative0 72.22
        superlative1 95.69
        superlative2 100.0
        superlative3 100.0
        yesno 89.0
```
#### v9
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* 添加 loss_results : score4 = (1 - gold_loss) * (1 - pred_loss)
    * aggregation
        * [(TODO1 & TODO3) & score3] : score4 = 1000 : 1
    * others
        * [(TODO1 & TODO3) & score3] : score4 = 2 : 1
* 添加覆盖词数score3
    * ( TODO1 & TODO3 ) : score3 = 400 : 1
* aggregation
    * TODO1 : TODO3 = 25 : 1
* single-relation
    * TODO1 : TODO3 = 1 : 8
* superlative0
    * TODO1 : TODO3 = 90 : 1
* multi-choice
    * TODO1 : TODO3 = 1 : 25
* others
    * TODO1 : TODO3 = 1 : 3

```
        overall 82.48
        aggregation 66.78
        comparative 100.0
        cvt 77.87
        multi-choice 48.51
        multi-constraint 87.03
        multi-hop 91.54
        multi-turn-answer 94.34
        multi-turn-entity 74.79
        multi-turn-predicate 88.0
        single-relation 88.6
        superlative 76.06
        superlative0 72.22
        superlative1 95.69
        superlative2 100.0
        superlative3 100.0
        yesno 89.0
```
#### v10
* 对于 **multi-turn-predicate** 和 **multi-turn-answer** ，不考虑分解的prediction
* 添加 loss_results : score4 = 1 - pred_loss
    * aggregation
        * [(TODO1 & TODO3) & score3] : score4 = 1000 : 1
    * others
        * [(TODO1 & TODO3) & score3] : score4 = 2 : 1
* 添加覆盖词数score3
    * ( TODO1 & TODO3 ) : score3 = 400 : 1
* aggregation
    * TODO1 : TODO3 = 25 : 1
* single-relation
    * TODO1 : TODO3 = 1 : 8
* superlative0
    * TODO1 : TODO3 = 90 : 1
* multi-choice
    * TODO1 : TODO3 = 1 : 25
* others
    * TODO1 : TODO3 = 1 : 3

```
        overall 82.29
        
        aggregation 66.78
        comparative 100.0
        cvt 77.71
        multi-choice 47.01
        multi-constraint 87.03
        multi-hop 91.54
        multi-turn-answer 94.34
        multi-turn-entity 73.97
        multi-turn-predicate 88.0
        single-relation 88.46
        superlative 76.17
        superlative0 72.35
        superlative1 95.69
        superlative2 100.0
        superlative3 100.0
        yesno 88.67
```
