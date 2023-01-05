# Advanced Deep Learning Final Project
- Topic: hahow user purchase prediction
- Team 26: RegressionIsAllYouNeed

- Team member
  - 資料科學碩一 李姵徵
  - 資工AI碩一 柳宇澤
  - 資工AI碩一 林聖硯
  - 網媒所碩一 何俞彥

## Topic predicion performance table

Metric: public seen / unseen mapk@50
|  Models / Features  |                      user-item                      |   text info.    |         text info + user-item interaction         |
| :-----------------: | :-------------------------------------------------: | :-------------: | :-----------------------------------------------: |
|       Hottest       |                   0.2085 / 0.1689                   |        -        |                         -                         |
|         UIF         |                          -                          |        -        |                         -                         |
|         MF          |                   0.0271 / 0.0238                   |        -        |                         -                         |
|         ALS         |                       0.0790                        |        -        |                         -                         |
|      Light GCN      |                   0.2706 / 0.0952                   |        -        |                  0.2289 / 0.2346                  |
| Logistic regression |                   0.0006 / 0.0005                   | 0.2717 / 0.2733 | 0.277 / <span style="color:red">**0.3104**</span> |
|      Ensemble       | <span style="color:red">**0.2812** </span> / 0.2347 |                 |                                                   |

## Course predicion performance table

Metric: public seen / unseen mapk@50
|  Models / Features  |                            user-item                            |   text info.    | text info + user-item interaction |
| :-----------------: | :-------------------------------------------------------------: | :-------------: | :-------------------------------: |
|       Hottest       |                         0.0359 / 0.0501                         |        -        |                 -                 |
|         UIF         |                         0.0363 / 0.0717                         |        -        |                 -                 |
|         MF          |                         0.0373 / 0.0281                         |        -        |                 -                 |
|         LMF         |                             0.0143                              |        -        |                 -                 |
|         ALS         |                             0.0476                              |        -        |                 -                 |
|      Light GCN      |                             0.03894                             |        -        |                 -                 |
| Logistic regression |                         0.0389 / 0.0170                         | 0.0465 / 0.0651 |         0.04664 / 0.09154         |
|      Ensemble       | <span style="color:red"> **0.05856 / 0.09173**          </span> |                 |                                   |

## Environment

```bash
conda env create -f env_logistic_regression.yml
conda env create -f env_lgn.yml
conda env create -f env_mf.yml
conda env create -f env_als.yml
```

## Downlaod files

```bash
bash log_reg_download.sh
bash lgn_download.sh
```

## Directory layout
```
ADL_FinalProject/ 
┣ ALS/
┣ LightGCN-PyTorch/ 
┣ MF/  
┣ ensemble/ 
┣ hahow/
┣ logistic_regression/
┣ notebooks/
┣ prediction/
┣ utils/
┣ .gitignore
┣ README.md
┣ als.sh
┣ als_reproduce.sh
┣ ensemble.sh
┣ env_als.yml
┣ env_lgn.yml
┣ env_logistic_regression.yml
┣ env_mf.yml
┣ lg_onehot_reproduce.sh
┣ lgn.sh
┣ lgn_download.sh
┣ lgn_reproduce.sh
┣ log_reg_download.sh
┣ log_reg_reproduce.sh
┣ logistic_regression.sh
┗ mf_reproduce.sh
```

## Reproduce best result of each competition
```bash
conda activate logistic_regression && bash logistic_regression.sh
conda activate lightGCN && bash lgn.sh
conda activate als && bash als.sh
bash ensemble.sh

kaggle competitions submit -c 2022-adl-final-hahow-unseen-user-topic-prediction -f ./prediction/log_reg_pred_unseen_topic.csv -m "Message"
kaggle competitions submit -c 2022-adl-final-hahow-seen-user-topic-prediction -f ./prediction/lgn0.4_lg0.6_seen_topic.csv -m "Message"
kaggle competitions submit -c 2022-adl-final-hahow-unseen-user-course-prediction -f ./prediction/lgn0.0_als0.5_lg0.5_unseen.csv -m "Message"
kaggle competitions submit -c 2022-adl-final-hahow-seen-user-course-prediction -f ./prediction/lgn0.025_als0.95_lg0.025_seen.csv -m "Message"
```

## Reproduce training and inference stage of each model 
### ALS
```bash
conda activate als && bash als_reproduce.sh
```

### Light GCN
```bash
# For more details, check `README.md` under `LightGCN-PyTorch`
conda activate lightGCN && bash lgn_reproduce.sh
```

### logistic regression (one-hot)
```bash
conda activate logistic_regression
bash lg_onehot_reproduce.sh
```

### logistic regression 
```bash
conda activate logistic_regression
bash log_reg_reproduce.sh
```

### Matrix factorization
```bash
conda activate MF
bash mf_reproduce.sh
```
