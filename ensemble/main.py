import argparse
import os
import numpy as np
import pandas as pd

from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble method')
    parser.add_argument('--task', type=str, default='course', help='course or topic (subgroup)')
    parser.add_argument('--seen', action='store_true')
    parser.add_argument('--save_path', type=str, default='./', help='save path')
    parser.add_argument('--lgn_weight', type=float, help='weight of LightGCN')
    parser.add_argument('--als_weight', type=float, help='weight of ALS')
    parser.add_argument('--lg_weight', type=float,  help='weight of Logistic Regression')

    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = parse_args()
    print(args)

    if not os.path.exists(f'{args.save_path}'):
        os.makedirs(f'{args.save_path}')

    if args.task == 'course':
        # load data
        if args.seen:
            test = pd.read_csv('./hahow/data/test/test_seen.csv').user_id.to_numpy()
        else:
            test = pd.read_csv('./hahow/data/test/test_unseen.csv').user_id.to_numpy()
        course_list = pd.read_csv('./hahow/data/courses.csv').course_id.to_numpy()

        lgn_course_score = load('./data/lgn_course_score.dict.pkl')
        als_course_score = load('./data/als_course_score.dict.pkl')
        lg_course_score = load('./data/lg_course_score.dict.pkl')
        print('Load course data successfully')

        # ensemble
        ensemble_course = {'user_id': [], 'course_id': []}
        for user in test:
            temp = []
            for course in course_list:
                score = 0
                more_weight = 0

                try:
                    score += lgn_course_score[(user, course)] * args.lgn_weight
                except:
                    more_weight += args.lgn_weight

                try:
                    score += lg_course_score[(user, course)] * args.lg_weight
                except:
                    more_weight += args.lg_weight

                score += als_course_score[(user, course)] * (args.als_weight + more_weight)

                temp.append(score)
            ensemble_course['user_id'].append(user)
            ensemble_course['course_id'].append(' '.join(course_list[np.argsort(np.array(temp))[::-1]][:50]))

        # save data
        if args.seen:
            pd.DataFrame(ensemble_course).to_csv(f'{args.save_path}lgn{args.lgn_weight}_als{args.als_weight}_lg{args.lg_weight}_seen.csv', index=False)
            print('Create seen course predict data successfully')
        else:
            pd.DataFrame(ensemble_course).to_csv(f'{args.save_path}lgn{args.lgn_weight}_als{args.als_weight}_lg{args.lg_weight}_unseen.csv', index=False)
            print('Create unseen course predict data successfully')

    elif args.task == 'topic':
        # load data
        if args.seen:
            test = pd.read_csv('../hahow/data/test/test_seen_group.csv').user_id.to_numpy()
        else:
            test = pd.read_csv('../hahow/data/test/test_unseen_group.csv').user_id.to_numpy()
        group_list = pd.read_csv('../hahow/data/subgroups.csv').subgroup_id.to_numpy()

        lgn_group_score = load('./data/lgn_group_score.dict.pkl')
        lg_group_score = load('./data/lg_group_score.dict.pkl')
        print('load group data successfully')

        # ensemble
        ensemble_group = {'user_id': [], 'subgroup': []}
        for user in test:
            temp = []
            for group in group_list:
                score = 0
                more_weight = 0
                
                try:
                    score += lg_group_score[(user, group)] * args.lg_weight
                except:
                    more_weight += args.lg_weight

                score += lgn_group_score[(user, group)] * (args.lgn_weight + more_weight)

                temp.append(score)
            ensemble_group['user_id'].append(user)
            ensemble_group['subgroup'].append(' '.join(map(str, group_list[np.argsort(np.array(temp))[::-1]])))

        # save data
        if args.seen:
            pd.DataFrame(ensemble_group).to_csv(f'{args.save_path}lgn{args.lgn_weight}_lg{args.lg_weight}_seen_topic.csv', index=False)
            print('Create seen topic predict data successfully')
        else:
            pd.DataFrame(ensemble_group).to_csv(f'{args.save_path}lgn{args.lgn_weight}_lg{args.lg_weight}_unseen_topic.csv', index=False)
            print('Create unseen topic predict data successfully')

    else:
        print('Task should be course or topic (subgroup)')
