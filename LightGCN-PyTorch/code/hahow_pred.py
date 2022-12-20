import world
import model
import utils
from my_utils.utils import *
from world import cprint
import register
from register import dataset
import torch
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
from scipy import spatial
import pickle
import os

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

if __name__ == '__main__':
    topk = 50

    u_id2name, u_name2id = get_remaps("user")
    c_id2name, c_name2id = get_remaps("course")
    g_id2name, g_name2id = get_remaps("g")
    seen, unseen = get_test_userlist(True), get_test_userlist(False)

    with torch.no_grad():
        weight_file = world.config['weight']
        name = os.path.basename(weight_file).replace(".pth.tar", ".csv")
        Recmodel = register.MODELS[world.model_name](world.config, dataset)
        Recmodel = Recmodel.to(world.device)
        Recmodel.load_state_dict(torch.load(os.path.join("./checkpoints", weight_file), map_location=torch.device('cpu')))
        Recmodel.eval()
        ratings = Recmodel.getAllUsersRating()

        t = "course" if "course" in name else "group"
        _topk = min(len(ratings[0]), topk)
        _, ratings_k = torch.topk(ratings, k=_topk)

        store_base = "./preds"
        write_preds(os.path.join(store_base, f"seen_{name}"), seen, ratings_k, t)
        write_preds(os.path.join(store_base, f"unseen_{name}"), unseen, ratings_k, t)