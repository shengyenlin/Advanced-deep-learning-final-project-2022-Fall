import pandas as pd
import numpy as np
import scipy.sparse as sparse
from argparse import ArgumentParser
import implicit
import pickle
import torch.nn.functional as F
import torch
import pickle
from pathlib import Path
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to train data.",
        default="../hahow/data/train/train.csv",
    )
    parser.add_argument(
        "--dict_path",
        type=Path,
        help="Path to generate dictionary",
        default="../ensemble/data/als_course_score.dict.pkl",
    )
    args = parser.parse_args()
    return args


def main(args):
    datapath = args.data_path
    df = pd.read_csv(datapath)
    
    with open('../utils/remap/course_name2id.pkl', 'rb') as f:
        course_name2id = pickle.load(f)
        num_of_course = len(course_name2id.keys())
        print(num_of_course)
    with open('../utils/remap/user_name2id.pkl', 'rb') as f:
        user_name2id = pickle.load(f)
        num_of_user = len(user_name2id.keys())
        print(num_of_user)
    with open('../utils/remap/user_id2name.pkl', 'rb') as f:
        user_id2name = pickle.load(f)
    with open('../utils/remap/course_id2name.pkl', 'rb') as f:
        course_id2name = pickle.load(f)
    R = np.zeros((num_of_user, num_of_course), dtype=np.int32)
    for index, row in df.iterrows():
        user_id =  user_name2id[row['user_id']]
        course_ids = row['course_id'].split()
        for course_id in course_ids:
            R[user_id][course_name2id[course_id]] = 1
    print(R.shape)
    sparse_user_course = sparse.csr_matrix(R)
    sparse_course_user = sparse.csr_matrix(R.T)
    print(sparse_user_course.shape)
    print(sparse_course_user.shape)
    #Building the model
    model = implicit.als.AlternatingLeastSquares(factors=25, regularization=0.1, iterations=400,calculate_training_loss=True,use_gpu=True)
    #factors=25, regularization=0.1, iterations=200,calculate_training_loss=True,use_gpu=True 0.0645
    #factors=25, regularization=0.1, iterations=400,calculate_training_loss=True,use_gpu=True 0.06549
    alpha_val = 40
    data_conf =  (sparse_user_course * alpha_val).astype('double')
    model.fit(data_conf)
    Dict = {}
   
   
   
    count = 0
    for user_name,user_id in user_name2id.items():
        count += 1
        # user_name = str(row['user_id'])
        # user_id = user_name2id[user_name]
        recommendations = model.recommend(user_id, sparse_user_course[user_id],N=num_of_course)
        courses = recommendations[0]
        probs = F.softmax(torch.tensor(recommendations[1]), dim=0)
        for course_id,prob in zip(courses,probs):
            course_name = str(course_id2name[course_id])
            Dict[(user_name,course_name)] = prob.item()
        print(count)
    print(len(Dict.keys()))
    pickle.dump(Dict, open(args.dict_path, 'wb'))
    return
   
if __name__ == '__main__':
    args = parse_args()
    main(args)