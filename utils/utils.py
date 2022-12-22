import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

################# General tool #################
import pickle

# quick read file
def o(path):
    return open(path, 'r')

# quick pickle I/O
def out_pkl(data, path):
    pickle.dump(data, open(path, 'wb'))
def load_pkl(path):
    return pickle.load(open(path, 'rb'))

################# Remap #################

def get_remaps(_type):
    """
    Parmeters: 
    _type: str - Selected remap type

    Return: 
    id2name: dict()
    name2id: dict()
    """
    _type = _type.lower()
    assert _type in ["course", "c", "subgroup", "g", "user", "u"], 'Available remap type: ["course", "c", "subgroup", "g", "user", "u"]'
    if _type == "c":
        _type = "course"
    elif _type == "g":
        _type = "subgroup"
    elif _type == "u":
        _type = "user"
    return load_pkl(os.path.join(dname, f"remap/{_type}_id2name.pkl")), load_pkl(os.path.join(dname, f"remap/{_type}_name2id.pkl"))

################# Prediction #################

pred_header = "user_id,course_id\n"
pred_gheader = "user_id,subgroup\n"

def get_test_userlist(seen=True):
    """
    Parameters:
    seen: bool - Select seen / unseen user list

    Return:
    user_list: list()
    """
    seen_or_not = "seen" if seen else "unseen"
    return load_pkl(os.path.join(dname, f"inference/{seen_or_not}_users_list.pkl"))

################# Hot #################

course2hot = load_pkl(os.path.join(dname, "hot/course2hot.pkl"))
group2hotcourses = load_pkl(os.path.join(dname, "hot/group2hotcourses.pkl"))