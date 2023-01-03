import argparse

from preprocess import *
from models import *
from utils import *

def parse_args():
	parser = argparse.ArgumentParser(description='ADL final project')
	parser.add_argument('--task', type=str, default='train', help='train: course, train_group: subgroup(topic)')
	parser.add_argument('--eval', type=str, default='seen', help='seen/unseen')
	parser.add_argument('--model', type=str, help='RecSys model: cf/mf')
	parser.add_argument('--cf_distance', type=str, default='cosine', help='cosine/pearson')
	parser.add_argument('--cf_user_based', action='store_true', default=True, help='user based: True, item based: False')
	parser.add_argument('--num_ns', type=int, default=2, help='number of negative sample')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	print(args)
	assert args.model != None, 'Model cannot be None'

	# Load Data
	train_data, user_col, item_col, item_data, user_data = load_data(args)
	val_data, test_data = load_eval_data(args.task, args.eval)

	# Collaborative Filtering
	if args.model == 'cf':
		new_train_data = create_surprise_data(train_data, user_col, item_col, args.num_ns, item_data)
		new_test_data = create_surprise_test(test_data, user_col, item_col, item_data)
		print('Create surprise data successfully')

		pred_data = cf_model(args, new_train_data, new_test_data)
		upload_data = create_upload_data(user_col, item_col, pred_data)
		ui = 'user' if args.cf_user_based else 'item'
		if args.task == 'train':
			group = ''
		else:
			group = '_group'
		upload_data.to_csv(f'./cf_{args.cf_distance}_{ui}_{args.eval}{group}.csv', index=False)
		print('Create predict data successfully')


	# Matrix Factorization
	elif args.model == 'mf':
		new_train_data = create_surprise_data(train_data, user_col, item_col, args.num_ns, item_data)
		new_test_data = create_surprise_test(test_data, user_col, item_col, item_data)
		print('Create surprise data successfully')

		pred_data = mf_model(args, new_train_data, new_test_data)
		upload_data = create_upload_data(user_col, item_col, pred_data)
		if args.task == 'train':
			group = ''
		else:
			group = '_group'
		upload_data.to_csv(f'./mf_{args.eval}{group}.csv', index=False)
		print('Create predict data successfully')


	else:
		print('models only have cf and mf')
