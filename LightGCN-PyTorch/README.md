
## LightGCN-pytorch
### Reproduce
#### Download
Download embeddings encoded by Sbert.
```bash
cd code
./download_sbert.sh
```

#### Train

```bash
cd code
./run_seen_course_train.sh
./run_unseen_course_train_sbert.sh
./run_seen_group_train.sh
./run_unseen_group_train_sbert.sh
```

#### Assemble score
Prepared for assemble / ensemble method.
```bash
cd code
./run_hahow_all_assemble_score.sh
```

### Prediction
Note that you need to change the dataset & hyperparms, respectively for **course** & **group**.
```bash
cd code
./run_hahow_pred.sh
```
