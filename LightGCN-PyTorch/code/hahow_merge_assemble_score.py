import pickle

def i_pkl(path):
    return pickle.load(open(path, 'rb'))

def o_pkl(f, path):
    pickle.dump(f, open(path, 'wb'))

course = {**i_pkl("./assemble/seen-course.dict.pkl"), **i_pkl("./assemble/unseen-course.dict.pkl")}
print(max(course.values()), min(course.values()))
group = {**i_pkl("./assemble/seen-group.dict.pkl"), **i_pkl("./assemble/unseen-group.dict.pkl")}
print(max(group.values()), min(group.values()))

o_pkl(course, "./assemble/lgn_course_score.dict.pkl")
o_pkl(group, "./assemble/lgn_group_score.dict.pkl")