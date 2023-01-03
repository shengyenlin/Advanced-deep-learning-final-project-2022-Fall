import os
import json
import time
import pickle
import datetime
import copy
import joblib
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from traditional_method.constants import DATA_ROOT
from traditional_method.user_embeddings import create_user_embed
from traditional_method.preprocess import remove_nan_in_group_df, make_multiple_hard_labels, process_gt_int_arr_to_str_arr, load_pkl, read_sbert_embed
from traditional_method.postprocess import process_pred_int_arr_to_str_arr_course, course_to_topic
from traditional_method.metrics import mapk
from traditional_method.utils import compute_cos_sim

RANDOM_SEED = 1234
TOP_K = 50
LABEL_COLUMN = 'course_id'
SUBGROUP_ID_TO_NAME_PATH = './hahow/data/subgroups.csv'
JSON_PATH = './utils/remap/courseid2subgroup.json'

############ load mapping ############
subgroup_df = pd.read_csv(SUBGROUP_ID_TO_NAME_PATH)
subgroupname2id = \
    dict(zip(subgroup_df['subgroup_name'], subgroup_df['subgroup_id']))
with open(JSON_PATH, 'r') as f:
    courseid2subgroup = json.load(f)
f.close()
######################################

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the training dataset.",
        default=DATA_ROOT
    )
    parser.add_argument(
        "--experiment_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./logistic_regression/traditional_method/experiment",
    )

    parser.add_argument(
        "--map_dir",
        type=Path,
        help="Directory to mapping files",
        default='../utils/remap'
    )
    parser.add_argument(
        "--sbert_dir",
        type=Path,
        default='../utils/sbert'
    )

    parser.add_argument(
        "--mode",
        type=str,
        help=['train, test'],
        default='train'
    )

    # inference settings
    parser.add_argument("--cache_dir", type=Path)
    parser.add_argument("--out_dir", type=Path, help="output directory during testing")
    parser.add_argument("--ft_path", type=Path, help="path to FastText bin")

    # data
    parser.add_argument("--remove_nan", action='store_true')
    parser.add_argument("--normalize_feature", action='store_true')
    parser.add_argument("--use_sbert_interaction", action='store_true')
    parser.add_argument("--cos_sim_interaction", action='store_true')
    parser.add_argument("--use_only_user_topic", action='store_true')

    # model
    parser.add_argument("--model", type=str, help=['logistic_regression, random forest, xgboost'])
    parser.add_argument("--normalize_pred_prob", action='store_true')
    parser.add_argument("--balanced_class_weight_in_logreg", action='store_true')
   
    args = parser.parse_args()
    return args

def user_proba_to_dict(pred_prob, user_list, id2coursename):
    out = dict()
    # outer loop: # user
    # inner loop: # course
    for i, user_prob_all in enumerate(pred_prob):
        for j, user_prob in enumerate(user_prob_all):
            # key: (user id, topic id (start from 1))
            key = (user_list[i], id2coursename[j])
            out[key] = user_prob
    return out

def predict_to_class_idx(clf, X):
    pred_prob = clf.predict_proba(X)
    y_pred = post_process_pred_prob(pred_prob)
    return y_pred, pred_prob

def decide_best_model(seen_mapk_record, unseen_mapk_record, param_grid):
    seen_best_param_idx = np.argmax(seen_mapk_record)
    unseen_best_param_idx = np.argmax(unseen_mapk_record)

    if seen_best_param_idx == unseen_best_param_idx:
        best_param = param_grid[seen_best_param_idx]
        print("Best params: ", best_param)
    else:
        best_param_seen = param_grid[seen_best_param_idx]
        best_param_unseen = param_grid[unseen_best_param_idx]
        print("Best params (seen): ", best_param_seen)
        print("Best params (unseen): ", best_param_unseen)

    return seen_best_param_idx, unseen_best_param_idx

def post_process_to_df(df, pred_list, col):
    preds = [' '.join(
            map(str, pred)
            ) for pred in pred_list]
    preds_df = pd.DataFrame(
        {
            'user_id': df['user_id'],
            str(col): preds
        }
    )
    return preds_df

def post_process_pred_prob(pred_prob):
    pred_class_sort = pred_prob.argsort()[:, ::-1]
    pred_top_50 = pred_class_sort[:, :50].tolist()
    return pred_top_50

def insert_data_train(user_id_list, purchase_record, X_empty, y_empty, userid2embed):
    for i, user_id in enumerate(user_id_list):
        X_empty[i, :] = userid2embed[user_id]
        y_empty[i] = purchase_record[i]        
    return X_empty, y_empty

def insert_data(user_id_list, X_empty, userid2embed):
    for i, user_id in enumerate(user_id_list):
        X_empty[i, :] = userid2embed[user_id]    
    return X_empty

def main(args):
    print("Start to do course prediction!")
    args_dict = vars(args)
    run_id = int(time.time())
    date = datetime.date.today().strftime("%m%d")
    print(f"Run id = {run_id}")
    if args.mode == 'train':
        experiment_dir = args.experiment_dir / 'course' / str(date) / str(run_id)
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)


    # read user info
    user_df = pd.read_csv(
        os.path.join(args.data_dir, 'users.csv')
    )

    # read train, valid, test file
    if args.mode == 'train':
        train_df = pd.read_csv(
            os.path.join(args.data_dir, 'train', 'train.csv')
        )

        val_seen_df = pd.read_csv(
            os.path.join(args.data_dir, 'val', 'val_seen.csv')
        )

        val_topic_seen_df = pd.read_csv(
            os.path.join(args.data_dir, 'val', 'val_seen_group.csv')
        )

        val_unseen_df = pd.read_csv(
            os.path.join(args.data_dir, 'val', 'val_unseen.csv')
        )

        val_topic_unseen_df = pd.read_csv(
            os.path.join(args.data_dir, 'val', 'val_unseen_group.csv')
        )
    
    

    test_seen_df = pd.read_csv(
        os.path.join(args.data_dir, 'test', 'test_seen.csv')
    )

    test_unseen_df = pd.read_csv(
        os.path.join(args.data_dir, 'test', 'test_unseen.csv')
    )

    # Remove nan in all dfs
    if args.remove_nan:
        if args.mode == 'train':
            train_df = remove_nan_in_group_df(train_df)
            val_seen_df = remove_nan_in_group_df(val_seen_df)
            val_topic_seen_df = remove_nan_in_group_df(val_topic_seen_df)
            val_unseen_df = remove_nan_in_group_df(val_unseen_df)
            val_topic_unseen_df = remove_nan_in_group_df(val_topic_unseen_df)
    test_seen_df = remove_nan_in_group_df(test_seen_df)
    test_unseen_df = remove_nan_in_group_df(test_unseen_df)

    # embedding preprocess
    userid2embed = dict()
    user_embed, _ = create_user_embed(user_df, args.ft_path)

    if args.use_sbert_interaction:
        # read sbert embed
        user_ids, user_sbert_embeds = read_sbert_embed(args.sbert_dir, 'user_name2embed.pkl')
        topic_names, topic_sbert_embeds = read_sbert_embed(args.sbert_dir, 'group_name2embed.pkl')
        course_names, course_sbert_embeds = read_sbert_embed(args.sbert_dir, 'course_name2embed.pkl')
        
        # compute interactions
        if args.cos_sim_interaction:
            user_course_interest = compute_cos_sim(user_sbert_embeds, course_sbert_embeds.T)
            user_topic_interest = compute_cos_sim(user_sbert_embeds, topic_sbert_embeds.T)
        else:
            user_course_interest = user_sbert_embeds @ course_sbert_embeds.T
            user_topic_interest = user_sbert_embeds @ topic_sbert_embeds.T

        if args.use_only_user_topic:
            user_embed = np.concatenate(
                [user_embed, user_topic_interest], axis = 1
            )
        else:
            # print(type(user_embed), type(user_course_interest), type(user_topic_interest))
            user_embed = np.concatenate(
                [user_embed, user_course_interest, user_topic_interest], axis = 1
            )

    for i, id in enumerate(user_df['user_id']):
        userid2embed[id] = user_embed[i]
    print("Finish embedding preprocessing")
    print(f"User embedding shape: {user_embed.shape}")

    # label preprocess
    if args.mode == 'train':
        purchase_record_users_id, purchase_record_labels = \
                make_multiple_hard_labels(train_df, LABEL_COLUMN)
        
        # map course id to idx
        course_names = np.unique(purchase_record_labels)
        ids = np.arange(len(course_names))
        # print(ids[:10])

        # start from 0
        id2coursename, coursename2id = \
            dict(zip(ids, course_names)), dict(zip(course_names, ids))

        # with open('id2coursename.pkl', 'wb') as f:
        #     pickle.dump(id2coursename, f)
        # with open('coursename2id.pkl', 'wb') as f:
        #     pickle.dump(coursename2id, f)
        # return 
        purchase_record_labels = [int(coursename2id[course_name]) for course_name in purchase_record_labels]
        print("Finish label preprocessing")

        # Training phase
        X_train, y_train = \
                np.zeros((len(purchase_record_users_id), user_embed.shape[1])),\
                np.zeros(len(purchase_record_users_id))


        X_train, y_train = insert_data_train(
            purchase_record_users_id, purchase_record_labels, 
            X_train, y_train, userid2embed
            )

        X_seen_val, X_unseen_val, y_seen_val, y_unseen_val = \
            np.zeros((val_seen_df.shape[0], user_embed.shape[1])), \
            np.zeros((val_unseen_df.shape[0], user_embed.shape[1])), \
            val_seen_df[LABEL_COLUMN].tolist(), \
            val_unseen_df[LABEL_COLUMN].tolist()

        y_topics_seen_val, y_topics_unseen_val = \
            val_topic_seen_df['subgroup'], val_topic_unseen_df['subgroup']
        
        # lists of lists
        y_seen_val, y_unseen_val, y_topics_seen_val, y_topics_unseen_val = \
            process_gt_int_arr_to_str_arr(y_seen_val), \
                process_gt_int_arr_to_str_arr(y_unseen_val), \
                    process_gt_int_arr_to_str_arr(y_topics_seen_val), \
                        process_gt_int_arr_to_str_arr(y_topics_unseen_val)
        
        X_seen_val = insert_data(
                val_seen_df['user_id'], 
                X_seen_val,
                userid2embed
                )

        X_unseen_val = insert_data(
                val_unseen_df['user_id'],
                X_unseen_val,
                userid2embed
                )   
    else:
        with open(args.cache_dir / 'id2coursename.pkl', 'rb') as f:
            id2coursename = pickle.load(f)
        with open(args.cache_dir / 'coursename2id.pkl', 'rb') as f:
            coursename2id = pickle.load(f)

    X_seen_test = np.zeros((test_seen_df.shape[0], user_embed.shape[1]))
    X_unseen_test = np.zeros((test_unseen_df.shape[0], user_embed.shape[1]))

    X_seen_test = insert_data(
            test_seen_df['user_id'], 
            X_seen_test,
            userid2embed
            )

    X_unseen_test = insert_data(
            test_unseen_df['user_id'],
            X_unseen_test,
            userid2embed
            )

    if args.normalize_feature:
        if args.mode == 'train':
            normalizer = StandardScaler()
            X_train = normalizer.fit_transform(X_train)
            X_seen_val = normalizer.transform(X_seen_val)
            X_unseen_val = normalizer.transform(X_unseen_val)
        else:
            normalizer = joblib.load(args.cache_dir / 'log_reg_train_normalizer.pkl')
        X_seen_test = normalizer.transform(X_seen_test)
        X_unseen_test = normalizer.transform(X_unseen_test)


    if args.mode == 'train':
        # Validation phase
        if args.model == 'logistic_regression':
            param_grid = [
                    # {
                    #     'penalty':[None]
                    # },
                    {
                    'C': [0.001, 0.01],
                    'penalty': ['l2']
                    },
                    {
                    # 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    # 'penalty': ['elasticnet'],
                    # 'l1_ratio': [0.1, 0.4, 0.7]
                    }
                ]

        elif args.model == 'random_forest':
            n_estimators = [int(x) for x in np.linspace(start = 300, stop = 1000, num = 5)]
            # Maximum number of levels inㄊ tree
            max_depth = [int(x) for x in np.linspace(5, 10, num = 6)]
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            param_grid = {'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'oob_score': [True]}
        elif args.model == 'xgboost':
            # params = {
            #     'min_child_weight': [1, 5, 10],
            #     'gamma': [0.5, 1, 1.5, 2, 5],
            #     'subsample': [0.6, 0.8, 1.0],
            #     'colsample_bytree': [0.6, 0.8, 1.0],
            #     'max_depth': [3, 4, 5]
            # }
            param_grid = {
                        'subsample': [0.6, 0.8, 1.0],
                        # 'min_child_weight': [1, 5, 10],
                        # 'gamma': [0.5, 1, 1.5, 2, 5],
                        'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5],
                        'max_depth': [3, 4, 5, 7],
                        'n_estimators': [150, 200, 300, 500],
                        # 'reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
                        # 'reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200]
                        }

        param_grid = ParameterGrid(param_grid)

        mapk_seen_courses_records = []
        mapk_unseen_courses_records = []
        mapk_seen_topics_records = []
        mapk_unseen_topics_records = []
        clfs = []

        print("="*50)
        print(f"Using {args.model} model")
        print(f"Use normalized feature: {args.normalize_feature}")
        print(f"Use balanced class weight in logistic regression: {args.balanced_class_weight_in_logreg}")
        print(f"Use Sbert interaction: {args.use_sbert_interaction}")
        print(f"Use only user x topic interaction: {args.use_only_user_topic}")
        print(f"Use only user x course interaction: {~args.use_only_user_topic}")
        print(f"Use cosine similarity interaction: {args.cos_sim_interaction}")
        for param in param_grid:
            print("Start training ...", param)
            if args.model == 'logistic_regression':
                clf = LogisticRegression(
                    random_state=RANDOM_SEED, 
                    max_iter = 2000,
                    class_weight='balanced' if args.balanced_class_weight_in_logreg else None,
                    verbose=0, **param
                    ).fit(X_train, y_train)

            elif args.model == 'random_forest':
                clf = RandomForestClassifier(
                    random_state = RANDOM_SEED,
                    # verbose=1,
                    n_jobs=-1,**param
                    ).fit(X_train, y_train)

            # elif args.model == 'xgboost':
            #     clf = xgb.XGBClassifier(
            #         random_state=RANDOM_SEED,
            #         objective="multi:softmax", 
            #         n_jobs=-1, **param
            #     ).fit(X_train, y_train, verbose=True)

            clfs.append(copy.deepcopy(clf))

            # list of list
            y_courses_seen_pred, _ = predict_to_class_idx(clf, X_seen_val)
            y_courses_unseen_pred, _ = predict_to_class_idx(clf, X_unseen_val)

            # print(y_courses_seen_pred)
            y_courses_seen_pred = process_pred_int_arr_to_str_arr_course(
                y_courses_seen_pred, id2coursename
                )
            y_courses_unseen_pred = process_pred_int_arr_to_str_arr_course(
                y_courses_unseen_pred, id2coursename
                )

            mapk_seen_courses = mapk(y_seen_val, y_courses_seen_pred, k=TOP_K)
            mapk_unseen_courses = mapk(y_unseen_val, y_courses_unseen_pred, k=TOP_K)
            mapk_seen_courses_records.append(mapk_seen_courses), mapk_unseen_courses_records.append(mapk_unseen_courses)

            y_topics_seen_pred = course_to_topic(y_courses_seen_pred, courseid2subgroup, subgroupname2id)
            y_topics_unseen_pred = course_to_topic(y_courses_unseen_pred, courseid2subgroup, subgroupname2id)

            mapk_seen_topics = mapk(y_topics_seen_val, y_topics_seen_pred, k=TOP_K)
            mapk_unseen_topics = mapk(y_topics_unseen_val, y_topics_unseen_pred, k=TOP_K)
            mapk_seen_topics_records.append(mapk_seen_topics), mapk_unseen_topics_records.append(mapk_unseen_topics)

            print(f"Courses MAP@50: seen {mapk_seen_courses} unseen {mapk_unseen_courses}")
            print(f"Topics MAP@50: seen {mapk_seen_topics} unseen {mapk_unseen_topics}")

        # TODO: try differnet class weight in loss function
        # TODO: try normalized probability
        
        # Decide best model
        seen_best_param_idx, unseen_best_param_idx = decide_best_model(
            mapk_seen_courses_records, mapk_unseen_courses_records, param_grid
        )

        seen_best_model = clfs[seen_best_param_idx]
        unseen_best_model = clfs[unseen_best_param_idx]

        joblib.dump(
            seen_best_model, 
            os.path.join(cache_dir, 'course_seen_model.joblib')
            )

        joblib.dump(
            unseen_best_model, 
            os.path.join(cache_dir, 'course_unseen_model.joblib')
            )

    else:
        seen_best_model = joblib.load(args.cache_dir / 'course_seen_model.joblib')
        unseen_best_model = joblib.load(args.cache_dir / 'course_unseen_model.joblib')        

    # TODO: store best params, seen and unseen mapk
    # testing
    print("Start testing!")

    y_seen_pred, y_seen_pred_prob = predict_to_class_idx(seen_best_model, X_seen_test)
    y_unseen_pred, y_unseen_pred_prob = predict_to_class_idx(unseen_best_model, X_unseen_test)

    seen_dict = user_proba_to_dict(y_seen_pred_prob, test_seen_df['user_id'], id2coursename)
    unseen_dict = user_proba_to_dict(y_unseen_pred_prob, test_unseen_df['user_id'], id2coursename)
    user_dict = {**seen_dict, **unseen_dict}

    # lists of lists
    y_courses_seen_pred = process_pred_int_arr_to_str_arr_course(y_seen_pred, id2coursename)
    y_courses_unseen_pred = process_pred_int_arr_to_str_arr_course(y_unseen_pred, id2coursename)

    y_topics_seen_pred = course_to_topic(y_courses_seen_pred, courseid2subgroup, subgroupname2id)
    y_topics_unseen_pred = course_to_topic(y_courses_unseen_pred, courseid2subgroup, subgroupname2id)

    courses_seen_df = post_process_to_df(test_seen_df, y_courses_seen_pred, col='course_id')
    courses_unseen_df = post_process_to_df(test_unseen_df, y_courses_unseen_pred, col='course_id')
    # topics_seen_df = post_process_to_df(test_seen_df, y_topics_seen_pred, col='subgroup')
    # topics_unseen_df = post_process_to_df(test_unseen_df, y_topics_unseen_pred, col='subgroup')

    ################## File output ##################
    if args.mode == 'test':
        cache_dir = args.out_dir
    courses_seen_df.to_csv(
        os.path.join(args.out_dir, 'log_reg_pred_seen_course.csv'), 
        index=False
        )

    courses_unseen_df.to_csv(
        os.path.join(args.out_dir, 'log_reg_pred_unseen_course.csv'), 
        index=False
        )

    # topics_seen_df.to_csv(
    #     os.path.join(cache_dir, 'topics_seen.csv'), 
    #     index=False
    #     )

    # topics_unseen_df.to_csv(
    #     os.path.join(cache_dir, 'topics_unseen.csv'), 
    #     index=False
    #     )

    with open(os.path.join(args.out_dir, 'lg_course_score.dict.pkl'), 'wb') as f:
        pickle.dump(user_dict, f)
    ################################################
    print("Finish course prediction")


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        args.experiment_dir.mkdir(parents=True, exist_ok=True)
    main(args)