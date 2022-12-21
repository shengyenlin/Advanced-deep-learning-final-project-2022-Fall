import os
import pickle
from typing import Dict

import pandas as pd

from .constants import DATA_ROOT, CACHE_ROOT

def read_pickle(map_path):
    f = open(map_path, 'wb')
    load_data = pickle.load(f)
    f.close()
    return load_data

def make_multiple_hard_labels(df_y, label_column):
    """

    Parameters
    ----------
        df: pd.DataFrame
            - should contain `user_id` and (`subgroup` or `course_id`) column
        label_column: str
            - should be `subgroup` or `course_id`

    Returns
    ----------
        users: list
            - contains user, one user may exist mulitple times, if he/she bought several courses / categories
        labels:
            - contains courses / categories that a user bought
    """

    users = []
    labels = []

    for _, data in df_y.iterrows():
        sub_label = data[label_column]
        user = data['user_id']

        # not nan
        if type(sub_label) == str:
            # user purchases multiple courses / categories
            if ' ' in sub_label:
                sub_label_split = sub_label.split(' ')

                for sub_label in sub_label_split:
                    users.append(user)
                    labels.append(int(sub_label))

            # user purchases one course / category
            else:
                users.append(user)
                labels.append(int(sub_label))
    
    return users, labels


def make_soft_label_for_each_record(df_y, label_column):
    # TODO: read course purchase record -> mapping course record to categories -> compute each categories ratio
    # PROBLEM: sklearn model can't use softlabel?
    pass
    # return users, soft_labels

