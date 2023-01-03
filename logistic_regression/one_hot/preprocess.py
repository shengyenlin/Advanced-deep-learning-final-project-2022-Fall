import pandas as pd

def user_onehot():
    users = pd.read_csv('../../hahow/data/users.csv')

    # gender
    gender_male, gender_female, gender_other = [], [], []
    for i in range(len(users.gender)):
        if users.gender[i] == 'male':
            gender_male.append(1)
            gender_female.append(0)
            gender_other.append(0)
            
        elif users.gender[i] == 'female':
            gender_male.append(0)
            gender_female.append(1)
            gender_other.append(0)
            
        elif users.gender[i] == 'other':
            gender_male.append(0)
            gender_female.append(0)
            gender_other.append(1)
            
        else:
            gender_male.append(0)
            gender_female.append(0)
            gender_other.append(0)
    user_gender = pd.DataFrame({'user_id': users.user_id, 'male': gender_male, 'female': gender_female, 'other': gender_other})

    # occupation
    occupation_titles_len = []
    occupation_tiltes_list = []
    for i in range(len(users.occupation_titles)):
        try:
            occupation_titles_len.append(len(users.occupation_titles[i].split(',')))
            occupation_tiltes_list.extend(users.occupation_titles[i].split(','))
        except:
            pass
    
    occupation_onehot_label = {}
    for key in list(set(occupation_tiltes_list)):
        occupation_onehot_label[f'occupation_{key}'] = [0] * len(users)

    for i in range(len(users.occupation_titles)):
        try:
            for occupation in users.occupation_titles[i].split(','):
                occupation_onehot_label[f'occupation_{occupation}'][i] = 1
        except:
            pass

    occupation_onehot_label['user_id'] = users.user_id
    user_occupation = pd.DataFrame(occupation_onehot_label)

    # interest
    interests_len = []
    category = []
    sub_category = []
    for i in range(len(users.interests)):
        try:
            interests_len.append(len(users.interests[i].split(',')))
            for interest in users.interests[i].split(','):
                category.append(interest[:interest.find('_')])
                sub_category.append(interest[interest.find('_')+1:])
        except:
            pass

    interest_onehot_label = {}
    for key in list(set(category)) + list(set(sub_category)):
        interest_onehot_label[f'category_{key}'] = [0] * len(users)

    for i in range(len(users.interests)):
        try:
            for interest in users.interests[i].split(','):
                group = interest[:interest.find('_')]
                subgroup = interest[interest.find('_')+1:]
                interest_onehot_label[f'category_{group}'][i] = 1
                interest_onehot_label[f'category_{subgroup}'][i] = 1
        except:
            pass

    interest_onehot_label['user_id'] = users.user_id
    user_interest = pd.DataFrame(interest_onehot_label)

    # recreation
    recreation_names_len = []
    recreation_names_list = []
    for i in range(len(users.occupation_titles)):
        try:
            recreation_names_len.append(len(users.recreation_names[i].split(',')))
            recreation_names_list.extend(users.recreation_names[i].split(','))
        except:
            pass

    recreation_onehot_label = {}
    for key in list(set(recreation_names_list)):
        recreation_onehot_label[f'recreation_{key}'] = [0] * len(users)

    for i in range(len(users.recreation_names)):
        try:
            for recreation in users.recreation_names[i].split(','):
                recreation_onehot_label[f'recreatioin_{recreation}'][i] = 1
        except:
            pass

    recreation_onehot_label['user_id'] = users.user_id
    user_recreation = pd.DataFrame(recreation_onehot_label)

    # merge
    new_users = user_gender.merge(user_occupation).merge(user_interest).merge(user_recreation)
    new_users.to_csv('./data/users_onehot.csv', index=False)  # save



def course_onehot():
    courses = pd.read_csv('../../hahow/data/courses.csv')

    # groups
    groups_len = []
    groups_list = []
    for i in range(len(courses.groups)):
        try:
            groups_len.append(len(courses.groups[i].split(',')))
            groups_list.extend(courses.groups[i].split(','))
        except:
            pass

    groups_onehot_label = {}
    for key in list(set(groups_list)):
        groups_onehot_label[f'groups_{key}'] = [0] * len(courses)

    for i in range(len(courses.groups)):
        try:
            for group in courses.groups[i].split(','):
                groups_onehot_label[f'groups_{group}'][i] = 1
        except:
            pass
    
    groups_onehot_label['course_id'] = courses.course_id
    course_groups = pd.DataFrame(groups_onehot_label)

    # sub_groups
    subgroups_len = []
    subgroups_list = []
    for i in range(len(courses.sub_groups)):
        try:
            subgroups_len.append(len(courses.sub_groups[i].split(',')))
            subgroups_list.extend(courses.sub_groups[i].split(','))
        except:
            pass

    subgroups_onehot_label = {}
    for key in list(set(subgroups_list)):
        subgroups_onehot_label[f'subgroups_{key}'] = [0] * len(courses)

    for i in range(len(courses.sub_groups)):
        try:
            for subgroup in courses.sub_groups[i].split(','):
                subgroups_onehot_label[f'subgroups_{subgroup}'][i] = 1
        except:
            pass

    subgroups_onehot_label['course_id'] = courses.course_id
    course_subgroups = pd.DataFrame(subgroups_onehot_label)

    # merge
    new_courses = course_groups.merge(course_subgroups)
    new_courses.to_csv('./data/courses_onehot.csv', index=False)  # save


if __name__ == '__main__':
    user_onehot()
    course_onehot()
    
