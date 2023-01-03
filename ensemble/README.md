# Ensemble method
Please put the data files in the folders below:
```
|- ADL_FinalProject
    |- hahow
        |- test_seen.csv
        |- test_unseen.csv
        |- test_seen_group.csv
        |- test_unseen_group.csv
        |- ...

    |- ensemble
        |- main.py
        |- utils.py
        |- run.sh
        |- ...
        |- data
            |- lgn_course_score.dict.pkl
            |- als_course_score.dict.pkl
            |- lg_course_score.dict.pkl
            |- lgn_group_score.dict.pkl
            |- lg_group_score.dict.pkl
    |- ...
```
            
## Inference
- seen course
```
python main.py \
    --task 'course' \
    --seen \
    --lgn_weight <weight_of_LightGCN> \
    --als_weight <weight_of_ALS> \
    --lg_weight <weight_of_LogisticRegression>
```

- unseen course
```
python main.py \
    --task 'course' \
    --lgn_weight <weight_of_LightGCN> \
    --als_weight <weight_of_ALS> \
    --lg_weight <weight_of_LogisticRegression>
```

- seen topic
```
python main.py \
    --task 'topic' \
    --seen \
    --lgn_weight <weight_of_LightGCN> \
    --lg_weight <weight_of_LogisticRegression>
```

- unseen topic
```
python main.py \
    --task 'topic' \
    --lgn_weight <weight_of_LightGCN> \
    --lg_weight <weight_of_LogisticRegression>
```

## Reproduce (best kaggle public score)
This will generate 3 csv files: 
    - `lgn0.025_als0.95_lg0.025_seen.csv` is for seen course competition.
    - `lgn0_als0.5_lg0.5_seen.csv` is for unseen course competition.
    - `lgn0.4_lg0.6_seen.csv` is for seen topic competition.

```
bash run.sh
```