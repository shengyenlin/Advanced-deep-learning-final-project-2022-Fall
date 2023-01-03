import pickle
import pandas as pd
import random
from tqdm import tqdm


if __name__ == '__main__':
    # read data
    course_chapter_items = pd.read_csv('../../hahow/data/course_chapter_items.csv')
    courses = pd.read_csv('../../hahow/data/courses.csv')
    subgroups = pd.read_csv('../../hahow/data/subgroups.csv')
    test_seen_group = pd.read_csv('../../hahow/data/test_seen_group.csv')
    test_seen = pd.read_csv('../../hahow/data/test_seen.csv')
    test_unseen_group = pd.read_csv('../../hahow/data/test_unseen_group.csv')
    test_unseen = pd.read_csv('../../hahow/data/test_unseen.csv')
    train_group = pd.read_csv('../../hahow/data/train_group.csv')
    train = pd.read_csv('../../hahow/data/train.csv')
    users = pd.read_csv('../../hahow/data/users.csv')

    new_users = pd.read_csv('./data/users_onehot.csv')
    new_items = pd.read_csv('./data/courses_onehot.csv')


    X = [col for col in new_users.columns if col != 'user_id']
    itemX = [col for col in new_items.columns if col != 'course_id']


    print('train')
    trainX = []
    trainY = []
    for i in tqdm(range(len(train))):
        for course in train.course_id[i].split(' '):
            trainX.append(new_users[X][new_users.user_id == train.user_id[i]].values[0])
            trainY.append(course)

    with open("./data/trainX", "wb") as fp:   # Pickling
        pickle.dump(trainX, fp)

    with open("./data/trainY", "wb") as fp:   # Pickling
        pickle.dump(trainY, fp)


    print('train_group')
    trainX_group = []
    trainY_group = []
    for i in tqdm(range(len(train_group))):
        try:
            for subgroup in train_group.subgroup[i].split(' '):
                trainX_group.append(new_users[X][new_users.user_id == train_group.user_id[i]].values[0])
                trainY_group.append(subgroup)
        except:
            pass
            
    with open("./data/trainX_group", "wb") as fp:   # Pickling
        pickle.dump(trainX_group, fp)

    with open("./data/trainY_group", "wb") as fp:   # Pickling
        pickle.dump(trainY_group, fp)


    print('test')
    testX_seen = []
    for i in tqdm(range(len(test_seen))):
        testX_seen.append(new_users[X][new_users.user_id == test_seen.user_id[i]].values[0])
        
    with open("./data/testX_seen", "wb") as fp:   # Pickling
        pickle.dump(testX_seen, fp)
        
        
    testX_unseen = []
    for i in tqdm(range(len(test_unseen))):
        testX_unseen.append(new_users[X][new_users.user_id == test_unseen.user_id[i]].values[0])
        
    with open("./data/testX_unseen", "wb") as fp:   # Pickling
        pickle.dump(testX_unseen, fp)

    
    print('test_group')
    testX_seen_group = []
    for i in tqdm(range(len(test_seen_group))):
        testX_seen_group.append(new_users[X][new_users.user_id == test_seen.user_id[i]].values[0])
        
    with open("./data/testX_seen_group", "wb") as fp:   # Pickling
        pickle.dump(testX_seen_group, fp)
        
        
    testX_unseen_group = []
    for i in tqdm(range(len(test_unseen_group))):
        testX_unseen_group.append(new_users[X][new_users.user_id == test_unseen.user_id[i]].values[0])
        
    with open("./data/testX_unseen_group", "wb") as fp:   # Pickling
        pickle.dump(testX_unseen_group, fp)