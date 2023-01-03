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

def write_rating(_store, ulist, rating, task):
    rlt = {}
    id2name = c_id2name if task == "course" else g_id2name
    for u in ulist:
        uid = u_name2id[u]
        r = rating[uid]
        #print("Min", min(r), "Max", max(r))
        for i, score in enumerate(r):
            item_name = id2name[i]
            if task == "group":
                item_name = int(item_name)
            rlt[(u, item_name)] = score.item()
    pickle.dump(rlt, open(_store, 'wb'))

if __name__ == '__main__':
    topk = 50

    u_id2name, u_name2id = get_remaps("user")
    c_id2name, c_name2id = get_remaps("course")
    g_id2name, g_name2id = get_remaps("g")
    seen, unseen = get_test_userlist(True), get_test_userlist(False)

    with torch.no_grad():
        weight_file = world.config['weight']
        name = os.path.basename(weight_file).replace(".pth.tar", ".dict.pkl")
        Recmodel = register.MODELS[world.model_name](world.config, dataset)
        Recmodel = Recmodel.to(world.device)
        Recmodel.load_state_dict(torch.load(os.path.join("./checkpoints", weight_file), map_location=torch.device('cpu')))
        Recmodel.eval()
        ratings = Recmodel.getAllUsersRating()

        t = "course" if "course" in name else "group"
        users = seen if world.config['users'] == "seen" else unseen
        users_str = "seen" if world.config['users'] == "seen" else "unseen"

        assemble_store_base = "./assemble"
        write_rating(os.path.join(assemble_store_base, f"{users_str}-{t}.dict.pkl"), users, ratings, t)