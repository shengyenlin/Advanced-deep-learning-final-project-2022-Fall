import os 
import pickle 

def process_pred_int_arr_to_str_arr_course(arr, id2coursename):
    arr_out = []
    for el in arr:
        arr_out.append([str(id2coursename[id]) for id in el])
    return arr_out 

def map_to_course(course_id_list, courseid2subgroup, subgroup_name_to_id):
    subgroups = list()
    for c_id in course_id_list:
        subgroup = courseid2subgroup[c_id]
        if type(subgroup) == list:
            for e in subgroup:
                subgroups.append(str(subgroup_name_to_id[e]))
        else:
            if subgroup == 'nan': #skip nan
                continue 
            subgroups.append(str(subgroup_name_to_id[subgroup]))

    return subgroups

def course_to_topic(courses_arrs, courseid2subgroup, subgroup_name_to_id):
    topic_idx_arrs = [
        map_to_course(course_arr, courseid2subgroup, subgroup_name_to_id) \
            for course_arr in courses_arrs
    ]

    topic_idx_arrs = [
        remove_duplicates(topic_arr) for topic_arr in topic_idx_arrs
    ]
    # print(topic_idx_arrs[0], topic_idx_arrs[1], topic_idx_arrs[2])
    return topic_idx_arrs

# remove duplicates while preserve order
def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]