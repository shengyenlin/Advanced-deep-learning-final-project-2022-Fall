# Advanced Deep Learning Final Project - hahow user purchase prediction
- Team 26: RegressionIsAllYouNeed
- Team member
  - 資料科學碩一 李姵徵
  - 資工AI碩一 柳宇澤
  - 資工AI碩一 林聖硯 Sheng-Yen Lin
  - 網媒所碩一 何俞彥

## Competition description
[Hahow](https://hahow.in/) is a Taiwanese online learning platform where you can purchase several courses that can help you build your future career skills. In this competition, we aim to use the historical purchase data of the website's members to predict which courses and course categories both current and new (cold-start) users are likely to buy.

## Competition result
- Oral presentation score: 9.7/10 (Ranked **1st** among all **61** teams)
- Report score: 20.5/26 (average=19.35)

Kaggle rank (Public / Private)
| [Unseen Topic](https://www.kaggle.com/competitions/2022-adl-final-hahow-unseen-user-topic-prediction/leaderboard?tab=public)  |                     [Seen Topic](https://www.kaggle.com/competitions/2022-adl-final-hahow-unseen-user-topic-prediction/team)                    |   [Unseen Course](https://www.kaggle.com/competitions/2022-adl-final-hahow-unseen-user-course-prediction/leaderboard) |         [Seen Course](https://www.kaggle.com/competitions/2022-adl-final-hahow-seen-user-course-prediction/submissions)         |
| :-----------------: | :-------------------------------------------------: | :-------------: | :-----------------------------------------------: |
|        23/21    |                   26/33                   |        28/26        |                         37/36                         |


## Brief intro to our findings

We used various recommendation methods for predicting user future purchase, including naive methods, traditional statistical methods, and deep learning methods. 

- Naive methods lack granularity but can serve as a baseline for comparison, while logistic regression performs well even with only user text information. initialized embeddings may be affected during updates for seen users. 
- The logistic regression model performs well on all recommendation tasks and incorporating user and course/topic interactions into the model significantly improves performance
- For deep learning methods, using deeper aggregation layers improves performance for sparser datasets, and initializing user and item embedding with sentence-bert vector helps with recommendations for **unseen users**. However, initialized embeddings may be affected during updates for **seen users**.
- An ensemble method combining regression, ALS, and LightGCN improves overall performance.

**Fr detailed experiment results, observations and our findings, please refer to `report.pdf`, and our [oral presentation](https://www.youtube.com/watch?v=UpOfI-Bp6pc).**

---

## Topic predicion performance table

Metric: public seen / unseen mapk@50
|  Models / Features  |                      user-item                      |   text info.    |         text info + user-item interaction         |
| :-----------------: | :-------------------------------------------------: | :-------------: | :-----------------------------------------------: |
|       Hottest       |                   0.2085 / 0.1689                   |        -        |                         -                         |
|         UIF         |                          -                          |        -        |                         -                         |
|         MF          |                   0.0271 / 0.0238                   |        -        |                         -                         |
|         ALS         |                       0.0790                        |        -        |                         -                         |
|      Light GCN      |                   0.2706 / 0.0952                   |        -        |                  0.2289 / 0.2369                  |
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
|      Light GCN      |                             0.03894                             |        -        |          0.0372 / 0.0670          |
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
bash als_download.sh
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

## Reproduce ensemble
```bash
conda activate logistic_regression
bash ensemble.sh

kaggle competitions submit -c 2022-adl-final-hahow-unseen-user-topic-prediction -f ./prediction/log_reg_pred_unseen_topic.csv -m "Message"
kaggle competitions submit -c 2022-adl-final-hahow-seen-user-topic-prediction -f ./prediction/lgn0.4_lg0.6_seen_topic.csv -m "Message"
kaggle competitions submit -c 2022-adl-final-hahow-unseen-user-course-prediction -f ./prediction/lgn0.0_als0.5_lg0.5_unseen.csv -m "Message"
kaggle competitions submit -c 2022-adl-final-hahow-seen-user-course-prediction -f ./prediction/lgn0.025_als0.95_lg0.025_seen.csv -m "Message"
```

## Reproduce ensemble models and ensemble
```bash
conda activate logistic_regression && bash logistic_regression.sh
conda activate lightGCN && bash lgn.sh
conda activate als && bash als.sh
conda activate logistic_regression && bash ensemble.sh

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
