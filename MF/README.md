# Matrix Factorization (MF)
Please put the data files in the folders below:
```
|- ADL_FinalProject
    |- hahow
        |- test_seen.csv
        |- test_unseen.csv
        |- test_seen_group.csv
        |- test_unseen_group.csv
        |- ...

    |- MF
        |- main.py
        |- models.py
        |- utils.py
        |- run.sh
        |- ...

    |- ...
```
## Training & Inference
```
python main.py \
    --model 'mf' \
    --task <train_or_train_group> \
    --eval <seen_or_unseen> \ 
    --num_ns <number_of_negative_samples>
```

## Reproduce (kaggle public score)
This will generate 4 csv files: 
- `mf_seen.csv` is for seen course competition.
- `mf_unseen.csv` is for unseen course competition.
- `mf_seen_group.csv` is for seen topic competition.
- `mf_unseen_group.csv` is for unseen topic competition.
```
bash run.sh
```