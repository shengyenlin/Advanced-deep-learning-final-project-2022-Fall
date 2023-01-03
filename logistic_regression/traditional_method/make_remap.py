import pickle

def o(path):
    return open(path, 'r')

def gen_mapping(f, name_index, name_rows=True):
    start_row = 1 if name_rows else 0
    id2name, name2id = {}, {}
    for id, l in enumerate(f.readlines()[start_row:]):
        ts = l.split(',')
        name = ts[name_index]
        id2name[id] = name
        name2id[name] = id
    return id2name, name2id

def out_pkl(data, path):
    pickle.dump(data, open(path, 'wb'))

if __name__ == "__main__":

    users_path = "../../hahow/data/users.csv"
    courses_path = "../../hahow/data/courses.csv"
    subgroups_path = "../../hahow/data/subgroups.csv"

    user_meta = o(users_path)
    course_meta = o(courses_path)
    subgroup_meta = o(subgroups_path)

    for s in ["user", "course", "subgroup"]:
        meta_f = globals()[f"{s}_meta"]
        id2name, name2id = gen_mapping(meta_f, 0)
        out_pkl(id2name, f"./{s}_id2name.pkl")
        out_pkl(name2id, f"./{s}_name2id.pkl")

