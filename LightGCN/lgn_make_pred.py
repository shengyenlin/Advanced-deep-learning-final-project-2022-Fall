from utils.utils import *
import torch
from torch import Tensor
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('-s', type=str)
    parser.add_argument('-l', '--layer', default=2, type=int,)
    parser.add_argument('-d', '--dim', default=128, type=int,)

    return parser.parse_args()

def write_preds(_store, ulist, rating_k, task):
    store = open(_store, 'w')
    header = pred_header if task == "course" else pred_gheader
    id2name = c_id2name if task == "course" else g_id2name
    store.write(header)
    for u in ulist:
        uid = u_name2id[u]
        r = rating_k[uid]
        items = map(lambda x: id2name[int(x)], r.tolist())
        store.write(f"{u},{' '.join(items)}\n")
    print(f"Succesfully wirte {_store} !")



if __name__ == "__main__":
    args = parse_args()

    tasks = ["course", "group"]
    topk = 100

    u_id2name, u_name2id = get_remaps("user")
    c_id2name, c_name2id = get_remaps("course")
    g_id2name, g_name2id = get_remaps("g")
    seen, unseen = get_test_userlist(True), get_test_userlist(False)

    for i, t in enumerate(tasks):
        user_path = f"../embeds/user_lgn-hahow_{t}-{args.l}-{args.d}.emb.pkl"
        item_path = f"../embeds/item_lgn-hahow_{t}-{args.l}-{args.d}.emb.pkl"
        user_embed = Tensor(load_pkl(user_path))
        item_embed = Tensor(load_pkl(item_path))
        ratings = torch.matmul(user_embed, item_embed.t())
        _topk = min(len(ratings[0]), topk)
        _, ratings_k = torch.topk(ratings, k=_topk)
        print(ratings_k.size())
        
        store_base = os.path.join("../outputs", args.s)
        if not os.path.isdir(store_base):
            os.mkdir(store_base)
        write_preds(os.path.join(store_base, f"seen_{t}.csv"), seen, ratings_k, t)
        write_preds(os.path.join(store_base, f"unseen_{t}.csv"), unseen, ratings_k, t)



