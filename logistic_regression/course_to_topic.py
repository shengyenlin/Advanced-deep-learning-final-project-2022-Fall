import json
import sys
import os

import numpy as np
import pandas as pd

JSON_PATH = './utils/remap/courseid2subgroup.json'
SUBGROUP_ID_TO_NAME_PATH = './hahow/data/subgroups.csv'

PRED_PATH = sys.argv[1]
PRED_UNSEEN_PATH = os.path.join(PRED_PATH, 'pred_unseen.csv')
PRED_SEEN_PATH = os.path.join(PRED_PATH, 'pred_seen.csv')

subgroup_df = pd.read_csv(SUBGROUP_ID_TO_NAME_PATH)
subgroup_name_to_id = dict(zip(subgroup_df['subgroup_name'], subgroup_df['subgroup_id']))

with open(JSON_PATH, 'r') as f:
    courseid2subgroup = json.load(f)
f.close()

def map_to_course(course_id_list):
    subgroups = list()
    for c_id in course_id_list.split(' '):
        subgroup = courseid2subgroup[c_id]
        if type(subgroup) == list:
            for e in subgroup:
                subgroups.append(str(subgroup_name_to_id[e]))
        else:
            if subgroup == 'nan': #skip nan
                continue 
            subgroups.append(str(subgroup_name_to_id[subgroup]))

    subgroups = ' '.join(subgroups)
    return subgroups


def main():
    # read course csv 
    pred_seen_course_df = pd.read_csv(PRED_SEEN_PATH)
    pred_unseen_course_df = pd.read_csv(PRED_UNSEEN_PATH)

    # map to category
    pred_seen_course_df['subgroup'] = \
        pred_seen_course_df['course_id'].apply(lambda x: map_to_course(x))
    pred_unseen_course_df['subgroup'] = \
        pred_unseen_course_df['course_id'].apply(lambda x: map_to_course(x))


    # TODO: add valid mapk

    pred_seen_subgroup_df = pred_seen_course_df[['user_id', 'subgroup']]
    pred_unseen_subgroup_df = pred_unseen_course_df[['user_id', 'subgroup']]

    pred_seen_subgroup_df.to_csv(
        os.path.join(PRED_PATH, 'pred_seen_topic.csv'), 
        index=False
        )

    pred_unseen_subgroup_df.to_csv(
        os.path.join(PRED_PATH, 'pred_unseen_topic.csv'), 
        index=False
        )

    # add to course_prediction

if __name__ == '__main__':
    main()