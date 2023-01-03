import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def load_data(args):
	train_data = pd.read_csv(f'../hahow/data/{args.task}.csv')
	user_data = pd.read_csv('../hahow/data/users.csv')
	user_col = 'user_id'

	if args.task.find('group') == -1:
		# course
		item_col = 'course_id'
		item_data = pd.read_csv('../hahow/data/courses.csv')
	else:
		# topic/subgroup
		item_col = 'subgroup_id'
		item_data = pd.read_csv('../hahow/data/subgroups.csv')

	if args.model == 'lg':
		user_data = pd.read_csv('../data/users_onehot.csv')
		item_data = pd.read_csv('../data/courses_onehot.csv')

	return train_data, user_col, item_col, item_data, user_data


def load_eval_data(task, seen):
	if task.find('group') == -1:
		task = ''
	else:
		task = '_group'

	val_data = pd.read_csv(f'../hahow/data/val_{seen}{task}.csv')
	test_data = pd.read_csv(f'../hahow/data/test_{seen}{task}.csv')
	return val_data, test_data


def create_upload_data(user_col, item_col, pred_data):
	upload_data = {user_col: [], item_col: []}
	for user in pred_data['user'].unique():
		upload_data[user_col].append(user)
		if item_col == 'subgroup_id':
			# topic
			upload_data[item_col].append(' '.join(map(str, list(pred_data.groupby('user').get_group(user).sort_values(by=['rating'], ascending=False)['item'][:50]))))
		else:
			# courses
			upload_data[item_col].append(' '.join(list(pred_data.groupby('user').get_group(user).sort_values(by=['rating'], ascending=False)['item'][:50])))

	return pd.DataFrame(upload_data)