import random
import pandas as pd

def create_surprise_data(data, user_col, item_col, num_ns, item_data):
    new_data = {'user': [], 'item': [], 'rating': []}
    if item_col == 'subgroup_id':
        item_col = 'subgroup'
    for user, item_list in zip(data[user_col], data[item_col]):
        try:
            new_data['user'].extend([user] * len(item_list.split()))
            new_data['item'].extend(item_list.split())
            new_data['rating'].extend([1] * len(item_list.split()))
        except:
            pass

        # negative sampling
        if item_col == 'subgroup':
            item_col = 'subgroup_id'
        try:
            ns_items = list(set(item_data[item_col]).difference(set(item_list.split())))
        except:
            ns_items = list(set(item_data[item_col]))
        random.shuffle(ns_items)
        new_data['user'].extend([user] * num_ns)
        new_data['item'].extend(ns_items[:num_ns])
        new_data['rating'].extend([0] * num_ns)

    return pd.DataFrame(new_data)


def create_surprise_test(data, user_col, item_col, item_data):
    new_data = {'user': [], 'item': [], 'rating': []}
    for user in data[user_col]:
        new_data['user'].extend([user] * len(set(item_data[item_col])))
        new_data['item'].extend(list(set(item_data[item_col])))
        new_data['rating'].extend([0] * len(set(item_data[item_col])))

    return pd.DataFrame(new_data)