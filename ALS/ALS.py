import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
import pickle
from argparse import ArgumentParser
from pathlib import Path
import sys 
sys.path.append("..") 
from utils.metrics import mapk
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=Path,
        help="Path to train data.",
        default="../hahow/data/train/train.csv",
    )
    parser.add_argument(
        "--val_file",
        type=Path,
        help="Path to val data.",
        default="../hahow/data/val/val_seen.csv",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to test data.",
        default="../hahow/data/test/test_seen.csv",
    )
    parser.add_argument(
        "--pred_file",
        type=str,
        help="Path to the pred file.",
        default="./pred.csv",
    )
    args = parser.parse_args()
    return args


def main(args):
    datapath = args.train_file
    test_path = args.test_file
    val_path = args.val_file
    df = pd.read_csv(datapath)
   
    with open('../utils/remap/course_name2id.pkl', 'rb') as f:
        course_name2id = pickle.load(f)
        num_of_course = len(course_name2id.keys())
        #print(num_of_course)
    with open('../utils/remap/user_name2id.pkl', 'rb') as f:
        user_name2id = pickle.load(f)
        num_of_user = len(user_name2id.keys())
        #print(num_of_user)
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
    # print(R.shape)
    sparse_user_course = sparse.csr_matrix(R)
    sparse_course_user = sparse.csr_matrix(R.T)
    # print(sparse_user_course.shape)
    # print(sparse_course_user.shape)
    #Building the model
    model = implicit.als.AlternatingLeastSquares(factors=25, regularization=0.1, iterations=800,calculate_training_loss=True,use_gpu=True)
    #factors=25, regularization=0.1, iterations=200,calculate_training_loss=True,use_gpu=True 0.0645
    #factors=25, regularization=0.1, iterations=400,calculate_training_loss=True,use_gpu=True 0.06549
    alpha_val = 100
    data_conf =  (sparse_user_course * alpha_val).astype('double')
    model.fit(data_conf)
  
    # Let's say we want to recommend artists for user with ID 7290
    # user_id = 7290
    # # recommend items for a user
    # recommendations = model.recommend(user_id, sparse_user_course[user_id],N=50)
    # print(recommendations)
    # user_id = 0
    # # recommend items for a user
    # recommendations = model.recommend(user_id, sparse_user_course[user_id],N=50)
    # print(recommendations)
    
   
    df_test = pd.read_csv(test_path)
    Pred = []
    GT = []
    #Validation
    df_val = pd.read_csv(val_path)
    for index,row in df_val.iterrows():
        user_name = row['user_id']
        user_id = user_name2id[user_name]
        recommendations = model.recommend(user_id, sparse_user_course[user_id],N=50)
        Pred.append(recommendations[0])
        gt = [course_name2id[i] for i in row['course_id'].split()]
        GT.append(gt)
    print(mapk(GT,Pred))
   
    #Testing
    with open(args.pred_file ,"w") as f:
        f.write("user_id,course_id\n")
        for index, row in df_test.iterrows():
            line = []
            user_name = row['user_id']
            user_id = user_name2id[user_name]
            recommendations = model.recommend(user_id, sparse_user_course[user_id],N=50)
            #print(recommendations[0])
            for rec in recommendations[0]:
                course_name = course_id2name[rec]
                line.append(course_name)
            l = " ".join(line)
            f.write(f"{user_name},{l}\n")

    return
   
if __name__ == '__main__':
    args = parse_args()
    main(args)