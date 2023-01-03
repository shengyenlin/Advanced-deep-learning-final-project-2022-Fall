# Logistic Regression - one hot encoding
Please put the data files in the folders below:
```
|- ADL_FinalProject
    |- hahow
        |- test_seen.csv
        |- test_unseen.csv
        |- test_seen_group.csv
        |- test_unseen_group.csv
        |- ...

    |- logistic_regression
        |- one_hot
            |- main.py
            |- preprocess.py
            |- utils.py
            |- run.sh
            |- data
            
        |- ...
    |- ...
```

## Preprocess
```
python preprocess.py
python create_data.py
```

## Training & Inference
```
python main.py
```

## Reproduce (kaggle public score)
This will generate 4 csv files: 
- `lg_onehot_seen.csv` is for seen course competition.
- `lg_onehot_unseen.csv` is for unseen course competition.
- `lg_onehot_seen_group.csv` is for seen topic competition.
- `lg_onehot_unseen_group.csv` is for unseen topic competition.
```
bash run.sh
```
