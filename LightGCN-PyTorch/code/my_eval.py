import world
import model
import utils
from world import cprint
import register
from register import dataset
import torch
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
from scipy import spatial

import pickle

def output_user_list(path):
    f = open(path, 'r')
    ls = f.read().strip().split('\n')[1:]
    id2name, name2id = {}, {}
    for l in ls:
        ts = l.strip().split(" ")
        name, id = ts[0], int(ts[1])
        id2name[id] = name
        name2id[name] = id
    return id2name, name2id

if __name__ == '__main__':
    id2name, name2id = output_user_list("./user_list.txt")
    pickle.dump(id2name, open("./outs/id2name.pkl", 'wb'))
    pickle.dump(name2id, open("./outs/name2id.pkl", 'wb'))
    
    weight_file = "checkpoints/lgn-suspether-2-64.pth.tar"
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
    user_embeds, n_user = Recmodel.embedding_user.weight.cpu().detach().numpy(), Recmodel.num_users
    item_embeds, n_item = Recmodel.embedding_item.weight.cpu().detach().numpy(), Recmodel.num_items
    pickle.dump(user_embeds, open("./outs/query_embeds.pkl", 'wb'))
    pickle.dump(item_embeds, open("./outs/addr_embeds.pkl", 'wb'))
    #assert n_user == n_item, f"{n_user}, {n_item}"
    print("# User: ", len(user_embeds), "# Item: ", len(item_embeds))
    answer_s = "0 1 2 3 4 4 3 3 2 4 4 4 4 4 4 5 6 7 8 3 9 10 11 11 12 2 12 13 12 14 15 16 17 15 18 18 8 19 19 16 20 16 21 21 22 21 22 2 22 23 22 22 22 22 22 24 3 25 26 10 7 27 28 28 3 28 3 3 15 3 18 29 10 23 29 3 29 2 29 3 3 3 15 3 3 30 8 2 31 15 32 33 33 33 15 3 15 34 34 15 3 2 35 7 36 15 35 33 37 33 15 23 3 2 35 38 3 39 8 10 40 39 8 40 40 40 10 40 39 15 8 41 42 2 43 41 7 2 44 3 8 44 3 45 10 3 10 10 10 10 18 10 10 10 7 10 3 10 10 44 3 2 46 10 3 47 48 8 2 48 2 48 49 3 48 50 27 49 48 3 51 15 51 8 48 18 7 52 52 10 10 53 23 2 54 54 18 8 2 8 2 3 10 55 55 10 3 3 56 57 58 58 58 58 58 58 58 58 58 58 58 58 58 59 27 10 60 61 2 61 10 62 57 8 23 8 63 64 15 2 64 65 66 67 3 15 64 68 69 64 70"
    answers = set(map(lambda x: int(x), answer_s.strip().split(' ')[1:]))
    tree = spatial.KDTree(item_embeds)
    rlt = tree.query(user_embeds[0], k=101)
    preds = rlt[1][1:]
    topks = [10, 20, 30, 40, 50, 100]
    for topk in topks:
        collap = len(set(preds[:topk]) & answers)
        print(f"Recall@{topk}: {collap/len(answers):4f}")
        print(f"Precision@{topk}: {collap/topk:4f}")
