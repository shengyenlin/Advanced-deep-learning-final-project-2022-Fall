import os
import math
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


import fasttext
import fasttext.util
fasttext.util.download_model('zh', if_exists='ignore')

from code.constants import DATA_ROOT
from code.user_embeddings import create_user_embed
from code.preprocess_label import make_multiple_hard_labels

def main():
    # read user info
    user_df = pd.read_csv(
        os.path.join(DATA_ROOT, 'users.csv')
    )

    # read train, valid, test file
    train_df = pd.read_csv(
        os.path.join(DATA_ROOT, 'train', 'train_group.csv')
    )
    
    val_seen_df = pd.read_csv(
        os.path.join(DATA_ROOT, 'val', 'val_seen_group.csv')
    )

    val_unseen_df = pd.read_csv(
        os.path.join(DATA_ROOT, 'val', 'val_unseen_group.csv')
    )

    test_seen_df = pd.read_csv(
        os.path.join(DATA_ROOT, 'test', 'test_seen_group.csv')
    )

    test_unseen_df = pd.read_csv(
        os.path.join(DATA_ROOT, 'test', 'test_unseen_group.csv')
    )

    # embedding preprocess
    userid2embed = dict()
    user_embed = create_user_embed(user_df)
    for i, id in enumerate(user_df['user_id']):
        userid2embed[id] = user_embed[i]
    print("Finish embedding preprocessing")

    # label preprocess
    purchase_record_users_id, purchase_record_labels = make_multiple_hard_labels(train_df, 'subgroup')
    print("Finish label preprocessing")
    
    # Training phase
    X_train, y_train = \
            np.zeros((len(purchase_record_users_id), user_embed.shape[1])), \
            np.zeros(len(purchase_record_users_id))

    for i, user_id in enumerate(purchase_record_users_id):
        X_train[i, :] = userid2embed[user_id]
        y_train[i] = purchase_record_labels[i]

    print(X_train.shape, y_train.shape)
    print("Check embedding result")
    print(X_train[:5, :10], y_train[:5])
    return 
    #TODO: try differnet class weight


    # Validation phase
    # TODO: load metrics for valid set
    # no need to change to hard label 

    # hyper parameter tuning & modeling on valid seen / unseen file
    
    # testing

    pass

if __name__ == '__main__':
    main()