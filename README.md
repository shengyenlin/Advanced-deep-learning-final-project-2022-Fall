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
|         MF          |                         0.0006 / 0.0281                         |        -        |                 -                 |
|         LMF         |                             0.0143                              |        -        |                 -                 |
|         ALS         |                             0.0476                              |        -        |                 -                 |
|      Light GCN      |                             0.03894                             |        -        |                 -                 |
| Logistic regression |                         0.0389 / 0.0170                         | 0.0465 / 0.0651 |         0.04664 / 0.09154         |
|      Ensemble       | <span style="color:red"> **0.05856 / 0.09173**          </span> |                 |                                   |

## Environment

## Downlaod files

```bash
# data
# preprocessing things
```

## Directory layout

## Reproduce best result of each competition
```bash
bash logistic_regression_inference.bash # -> dictionary to ./ensemble
bash light_GCN_inference.bash
bash ensemble.bash
```


## Reproduce inference of each model
```bash
bash logistic_regression.bash # -> dictionary to ./ensemble

```
## Reproduce training of each model 

### Logistic regression
```bash
bash log_reg_topic.bash # reproduce topic prediction using logistic regression
bash log_reg_course.bash # reproduce course prediction using logistic regression
```

## Shared data
- [Google Cloud - shared data](https://drive.google.com/drive/folders/1g16qbUM4daiEbHD-JbtEMerF3-LfNkPm?usp=share_link)
