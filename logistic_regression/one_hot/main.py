import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
	# read data
	with open("./data/trainX", "rb") as fp:   # Unpickling
		trainX = pickle.load(fp)

	with open("./data/trainY", "rb") as fp:   # Unpickling
		trainY = pickle.load(fp)

	with open("./data/trainX_group", "rb") as fp:   # Unpickling
		trainX_group = pickle.load(fp)
	
	with open("./data/trainY_group", "rb") as fp:   # Unpickling
		trainY_group = pickle.load(fp)

	with open("./data/testX_seen", "rb") as fp:   # Unpickling
		testX_seen = pickle.load(fp)

	with open("./data/testX_seen_group", "rb") as fp:   # Unpickling
		testX_seen_group = pickle.load(fp)

	with open("./data/testX_unseen", "rb") as fp:   # Unpickling
		testX_unseen = pickle.load(fp)

	with open("./data/testX_unseen_group", "rb") as fp:   # Unpickling
		testX_unseen_group = pickle.load(fp)


	# course
	clf = LogisticRegression().fit(trainX, trainY)
	testY_seen = clf.predict_proba(testX_seen)
	testY_unseen = clf.predict_proba(testX_unseen)
	
	top50 = np.array(clf.classes_)[np.argsort(testY_seen)[::-1][:, :50]]
	test_seen = pd.read_csv('../../hahow/data/test_seen.csv')
	pred = {'user_id': [], 'course_id': []}
	for i in range(len(test_seen)):
		pred['user_id'].append(test_seen.user_id[i])
		pred['course_id'].append(' '.join(top50[i]))
	pd.DataFrame(pred).to_csv('./lg_onehot_seen.csv', index=False)

	top50 = np.array(clf.classes_)[np.argsort(testY_unseen)[::-1][:, :50]]
	test_unseen = pd.read_csv('../../hahow/data/test_unseen.csv')
	pred = {'user_id': [], 'course_id': []}
	for i in range(len(test_unseen)):
		pred['user_id'].append(test_unseen.user_id[i])
		pred['course_id'].append(' '.join(top50[i]))
	pd.DataFrame(pred).to_csv('./lg_onehot_unseen.csv', index=False)


	# group
	clf = LogisticRegression().fit(trainX_group, trainY_group)
	testY_seen_group = clf.predict_proba(testX_seen_group)
	testY_unseen_group = clf.predict_proba(testX_unseen_group)

	top50 = np.array(clf.classes_)[np.argsort(testY_seen_group)[::-1][:, :50]]
	test_seen_group = pd.read_csv('../../hahow/data/test_seen_group.csv')
	pred = {'user_id': [], 'subgroup': []}
	for i in range(len(test_seen_group)):
		pred['user_id'].append(test_seen_group.user_id[i])
		pred['subgroup'].append(' '.join(top50[i]))
	pd.DataFrame(pred).to_csv('./lg_onehot_seen_group.csv', index=False)

	top50 = np.array(clf.classes_)[np.argsort(testY_unseen_group)[::-1][:, :50]]
	test_unseen_group = pd.read_csv('../../hahow/data/test_unseen_group.csv')
	pred = {'user_id': [], 'subgroup': []}
	for i in range(len(test_unseen_group)):
		pred['user_id'].append(test_unseen_group.user_id[i])
		pred['subgroup'].append(' '.join(top50[i]))
pd.DataFrame(pred).to_csv('./lg_onehot_unseen_group.csv', index=False)

