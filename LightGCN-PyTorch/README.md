
## LightGCN-pytorch
### Environment
Please make sure `python == 3.7` then run:
```bash
pip install -r requirements.txt
```
### Reproduce
#### Download
Download embeddings encoded by Sbert.
```bash
pushd code; ./download_sbert.sh; popd
```

#### Train
Models will be stored in `checkpoints`.
```bash
pushd code; ./run_seen_course_train.sh; ./run_unseen_course_train_sbert.sh; ./run_seen_group_train.sh; ./run_unseen_group_train_sbert.sh; popd
```

#### Assemble score
Prepared for assemble / ensemble method.
```bash
pushd code; ./run_hahow_all_assemble_score.sh; popd
```

### Prediction
Predictions will be stored in `preds`
#### From scratch
Note that if you trained with different settings, you'll need to change the dataset & hyperparms in `run_hahow_pred.sh`.
```bash
pushd code; ./run_hahow_pred.sh; popd
```
#### From pretrained
```bash
pushd ..; ./lgn_download.sh; popd # You will see ../models if you successfully download them
pushd code; ./run_hahow_pred_pretrained.sh; popd
```
