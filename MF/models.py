from surprise import KNNBasic, SVD, Reader, Dataset

def cf_model(args, train_data, test_data):
	algo = KNNBasic(sim_options={'name': args.cf_distance,
								 'user_based': args.cf_user_based})

	reader = Reader(rating_scale=(0, 1))
	trainset = Dataset.load_from_df(train_data[['user', 'item', 'rating']], reader)
	algo.fit(trainset.build_full_trainset())

	for i in range(test_data.shape[0]):
		test_data['rating'][i] = algo.predict(test_data['user'][i], test_data['item'][i])
	return test_data


def mf_model(args, train_data, test_data):
	algo = SVD()
	reader = Reader(rating_scale=(0, 1))
	trainset = Dataset.load_from_df(train_data[['user', 'item', 'rating']], reader)
	algo.fit(trainset.build_full_trainset())

	for i in range(test_data.shape[0]):
		test_data['rating'][i] = algo.predict(test_data['user'][i], test_data['item'][i])
	return test_data