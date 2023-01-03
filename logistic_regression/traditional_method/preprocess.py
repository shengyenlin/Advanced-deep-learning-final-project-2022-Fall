import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd

from .constants import DATA_ROOT, CACHE_ROOT

def load_pkl(path):
    with open(path, 'rb') as f:
        pkl_file = pickle.load(f)
    return pkl_file

def convert_embed_dict_to_np_arr(embeds, dim_sbert=768):
    embeds_np = np.zeros((len(embeds), dim_sbert))
    for i, embed in enumerate(embeds.values()):
        embeds_np[i, :] = embed
    return embeds_np

def read_sbert_embed(sbert_path, pkl_name):
    embed_path = os.path.join(sbert_path, pkl_name)
    embeds = load_pkl(embed_path)
    idx = list(embeds.keys())
    embeds_np = convert_embed_dict_to_np_arr(embeds)
    return idx, embeds_np

def remove_nan_in_group_df(df:pd.DataFrame):
    df = df.dropna()
    return df

def process_gt_int_arr_to_str_arr(arr):
    arr_out = []
    for el in arr:
        el = el.split(' ')
        arr_out.append([str(e) for e in el])
    return arr_out

def process_pred_int_arr_to_str_arr_topic(arr):
    arr_out = []
    for el in arr:
        arr_out.append([str(e) for e in el])
    return arr_out 

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
                    if label_column == 'course_id':
                        labels.append(str(sub_label))
                    elif label_column == 'subgroup':
                        labels.append(int(sub_label))

            # user purchases one course / category
            else:
                users.append(user)
                if label_column == 'course_id':
                    labels.append(str(sub_label))
                elif label_column == 'subgroup':
                    labels.append(int(sub_label))
    
    return users, labels


def make_soft_label_for_each_record(df_y, label_column):
    # TODO: read course purchase record -> mapping course record to categories -> compute each categories ratio
    # PROBLEM: sklearn model can't use softlabel?
    pass
    # return users, soft_labels

