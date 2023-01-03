import os
import time
import datetime
import copy
import joblib
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd

# models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch

import fasttext
import fasttext.util
fasttext.util.download_model('zh', if_exists='ignore')

from traditional_method.constants import DATA_ROOT
from traditional_method.user_embeddings import create_user_embed
from traditional_method.preprocess import remove_nan_in_group_df, make_multiple_hard_labels, process_gt_int_arr_to_str_arr, process_pred_int_arr_to_str_arr_topic, load_pkl, read_sbert_embed
from traditional_method.metrics import mapk

from NN.dataset import UserDataset
from NN.model import DNN

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

RANDOM_SEED = 1234
TOP_K = 50

print("Random Seed:", SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the training dataset.",
        default=DATA_ROOT,
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./traditional_method/cache",
    )
    parser.add_argument(
        "--sbert_dir",
        type=Path,
        default='./utils/sbert'
    )

    # data
    parser.add_argument("--normalize_feature", action='store_true')
    parser.add_argument("--use_sbert_interaction", action='store_true')

    ## nn
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_worker", type=int, default=8)

    # model
    parser.add_argument("--model", type=str, help=['logistic regression, random forest, xgboost, knn'])
    parser.add_argument("--normalize_pred_prob", action='store_true')
    parser.add_argument("--balanced_class_weight_in_logreg", action='store_true')
    args = parser.parse_args()
    return args



def predict_to_class(clf, X, normalized_probability=False, global_prob=None):
    pred_prob = clf.predict_proba(X)
    if normalized_probability:
        pred_prob = pred_prob / global_prob[np.newaxis, :]
    y_pred = post_process_pred_prob(pred_prob)
    return y_pred

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

def post_process_label_to_df(df, pred_list):
    preds = [' '.join(
            map(str, pred)
            ) for pred in pred_list]
    preds_df = pd.DataFrame(
        {
            'user_id': df['user_id'],
            'subgroup': preds
        }
    )
    return preds_df

def post_process_pred_prob(pred_prob, fix_class_id=True):
    pred_class_sort = pred_prob.argsort()[:, ::-1]
    if fix_class_id:
        pred_top_50 = pred_class_sort[:, :50] + 1
    else:
        pred_top_50 = pred_class_sort[:, :50]
    return pred_top_50.tolist()

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
    args_dict = vars(args)
    run_id = int(time.time())
    date = datetime.date.today().strftime("%m%d")
    print(f"Run id = {run_id}")
    cache_dir = args.cache_dir / str(date) / str(run_id)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

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

    # Remove nan in all dfs
    train_df = remove_nan_in_group_df(train_df)
    val_seen_df = remove_nan_in_group_df(val_seen_df)
    val_unseen_df = remove_nan_in_group_df(val_unseen_df)
    test_seen_df = remove_nan_in_group_df(test_seen_df)
    test_unseen_df = remove_nan_in_group_df(test_unseen_df)

    # embedding preprocess
    userid2embed = dict()
    user_embed = create_user_embed(user_df)
    
    if args.use_sbert_interaction:
        # read sbert embed
        user_ids, user_sbert_embeds = read_sbert_embed(args.sbert_dir, 'user_name2embed.pkl')
        topic_names, topic_sbert_embeds = read_sbert_embed(args.sbert_dir, 'group_name2embed.pkl')
        course_names, course_sbert_embeds = read_sbert_embed(args.sbert_dir, 'course_name2embed.pkl')
        
        # compute interactions
        user_course_interest = user_sbert_embeds @ course_sbert_embeds.T
        user_topic_interest = user_sbert_embeds @ topic_sbert_embeds.T

        user_embed = np.concatenate(
            [user_embed, user_course_interest, user_topic_interest], axis = 1
            # [user_embed, user_topic_interest], axis = 1
        )

    for i, id in enumerate(user_df['user_id']):
        userid2embed[id] = user_embed[i]

    print("Finish embedding preprocessing")
    print(f"User embedding shape: {user_embed.shape}")

    # label preprocess
    purchase_record_users_id, purchase_record_labels = make_multiple_hard_labels(train_df, 'subgroup')
    purchase_record_labels = pd.Series(purchase_record_labels)
    global_class_prob = (purchase_record_labels.value_counts() / purchase_record_labels.shape).sort_index().to_numpy()
    print("Finish label preprocessing")
     
    # Training phase
    X_train, y_train = \
            np.zeros((len(purchase_record_users_id), user_embed.shape[1])), \
            np.zeros(len(purchase_record_users_id))


    X_train, y_train = insert_data_train(
        purchase_record_users_id, purchase_record_labels, 
        X_train, y_train, userid2embed
        )

    X_seen_val, X_unseen_val, y_seen_val, y_unseen_val = \
        np.zeros((val_seen_df.shape[0], user_embed.shape[1])), \
        np.zeros((val_unseen_df.shape[0], user_embed.shape[1])), \
        val_seen_df['subgroup'].tolist(), \
        val_unseen_df['subgroup'].tolist()
    
    y_seen_val, y_unseen_val = \
        process_gt_int_arr_to_str_arr(y_seen_val), \
            process_gt_int_arr_to_str_arr(y_unseen_val)
    
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
        normalizer = StandardScaler()
        X_train = normalizer.fit_transform(X_train)
        X_seen_val = normalizer.transform(X_seen_val)
        X_unseen_val = normalizer.transform(X_unseen_val)
        X_seen_test = normalizer.transform(X_seen_test)
        X_unseen_test = normalizer.transform(X_unseen_test)

    train_ds = UserDataset(X_train, y_train)
    val_seen_ds = UserDataset(X_seen_val, y_seen_val)
    val_unseen_ds = UserDataset(X_unseen_val, y_unseen_val)
    test_seen_ds = UserDataset(X_seen_test)
    test_unseen_ds = UserDataset(X_unseen_test)

    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    val_seen_dl = torch.utils.data.DataLoader(
        val_seen_ds, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    val_unseen_dl = torch.utils.data.DataLoader(
        val_unseen_ds, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_seen_dl = torch.utils.data.DataLoader(
        test_seen_ds, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    test_unseen_dl = torch.utils.data.DataLoader(
        test_unseen_ds, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    loss_fn = torch.nn.CrossEntropyLoss() 
    num_class = len(np.unique(y_train))
    model = DNN(
        in_dim = X_train.shape[1],
        out_dim = num_class
    )
    model = model.to(device)

    iters = 0
    for epoch in range(args.num_epoch):
        for i, X_train, y_train in enumerate(tqdm(train_dl, leave=False, colour='green')):
            hist['epoch'].append(epoch)
            hist['iter'].append(i)
            train_batch(data, netD, netG, optimizerD, optimizerG, loss_fn, args, hist)
            
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (hist['epoch'][-1], args.num_epoch, i, len(train_dl),
                        hist['D_losses'][-1], hist['G_losses'][-1], 
                        hist['D(x)'][-1], hist['D(G(z1))'][-1], hist['D(G(z2))'][-1]),
                        end = ' '
                        )

                eval(fixed_noise, netD, netG, epoch, i, img_save_path_ind, img_save_path_grid, hist)
                #store best epoch, show in the end at store as training result
                save_best(netD, netG, best, hist, model_save_path)
                gc.collect()
                torch.cuda.empty_cache()

                # Record every 50 iters
                with open(Path(img_save_path) / 'best.txt', 'w') as f:
                    print(best, file=f)
                    print(args_dict, file=f)
                f.close()

                
            iters += 1
            gc.collect()
            torch.cuda.empty_cache()

    #TODO: cmd line: ctrl + L OR clear tqdm output
    print(f"Finish model training, best training result:")
    print(best)
    plot_eval_graph(hist, img_save_path)

    with open(Path(img_save_path) / 'best.txt', 'w') as f:
        print(best, file=f)
        print(args_dict, file=f)
    f.close()

    best_save_path = os.path.join(img_save_path, 'best')
    os.mkdir(best_save_path)
    generate_pics_with_best_model(
        os.path.join(model_save_path, 'best.ckpt'), 
        best_save_path, fixed_noise, args)

    print("="*50)
    print(f"Using {args.model} model")
    print(f"Use normalized feature: {args.normalize_feature}")
    print(f"Use user topic course sbert interaction: {args.use_sbert_interaction}")

    y_seen_pred = process_pred_int_arr_to_str_arr_topic(y_seen_pred)
    y_unseen_pred = process_pred_int_arr_to_str_arr_topic(y_unseen_pred)

    mapk_seen = mapk(y_seen_val, y_seen_pred, k=TOP_K)
    mapk_unseen = mapk(y_unseen_val, y_unseen_pred, k=TOP_K)
    seen_mapk_records.append(mapk_seen), unseen_mapk_records.append(mapk_unseen)
    print(f"MAP@50: seen {mapk_seen} unseen {mapk_unseen}")

    # TODO: try differnet class weight in loss function
    
    # Decide best model
    seen_best_param_idx, unseen_best_param_idx = decide_best_model(
        seen_mapk_records, unseen_mapk_records, param_grid
    )

    print("Finish hyperparmeter tuning!")
    print("Seen best param: ", param_grid[seen_best_param_idx])
    print("Uneen best param: ", param_grid[unseen_best_param_idx])

    seen_best_model = clfs[seen_best_param_idx]
    unseen_best_model = clfs[unseen_best_param_idx]

    joblib.dump(
        seen_best_model, 
        os.path.join(args.cache_dir, 'category_seen_model.joblib')
        )

    joblib.dump(
        unseen_best_model, 
        os.path.join(args.cache_dir, 'category_unseen_model.joblib')
        )

    # TODO: store best params, seen and unseen mapk

    # testing
    print("Start testing!")
    y_seen_pred = predict_to_class(
        seen_best_model, X_seen_test, 
        args.normalize_pred_prob, global_class_prob
        )
    y_unseen_pred = predict_to_class(
        unseen_best_model, X_unseen_test, 
        args.normalize_pred_prob, global_class_prob
        )

    seen_df = post_process_label_to_df(test_seen_df, y_seen_pred)
    unseen_df = post_process_label_to_df(test_unseen_df, y_unseen_pred)

    ################## To csv ########################
    seen_df.to_csv(
        os.path.join(cache_dir, 'pred_seen_topic.csv'), 
        index=False
        )

    unseen_df.to_csv(
        os.path.join(cache_dir, 'pred_unseen_topic.csv'), 
        index=False
        )
    ################################################

if __name__ == '__main__':
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    main(args)