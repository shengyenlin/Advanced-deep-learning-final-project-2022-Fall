import pickle
import os

import numpy as np

SBERT_PATH = '../utils/sbert'

def main():
    course_embed_path = os.path.join(SBERT_PATH, 'course_name2embed.pkl')
    user_embed_path = os.path.join(SBERT_PATH, 'user_name2embed.pkl')
    with open(course_embed_path, 'rb') as f:
        course_embeds = pickle.load(f)

    with open(user_embed_path, 'rb') as f:
        user_embeds = pickle.load(f)

    course_embed_np = np.zeros((len(course_embeds), 768))
    user_embed_np = np.zeros((len(user_embeds), 768))

    for i, course_embed in enumerate(course_embeds.values()):
        course_embed_np[i, :] = course_embed

    for i, user_embed in enumerate(user_embeds.values()):
        user_embed_np[i, :] = user_embed
    
    user_course_interest = user_embed_np @ course_embed_np.T
    # (# user, # course)
    print(user_course_interest.shape)



if __name__ == '__main__':
    main()