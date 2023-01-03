import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer

import fasttext
import fasttext.util

def load_ft_model(ft_path):
    ft = fasttext.load_model(str(ft_path))
    return ft

def create_gender_embbeding(user_df):
    """
    Compute gender embedding.

    Parameters
    ----------
    user_df : pd.DataFrame
        - Should contains `gender` column

    Returns
    -------
    gender_embed: np.array
        - One hot embedding of `gender` column
        - shape: (# data, # of gender)
    """
    gender_embed = pd.get_dummies(user_df['gender']).to_numpy()
    return gender_embed

def create_occupation_embed(user_df, ft, embed_size=300):
    """
    Compute occpuation embedding.

    Parameters
    ----------
    user_df : pd.DataFrame
        - Should contains `occupation_titles` column
    ft: fasttext word vector model
    embed_size: size of word embedding

    Returns
    -------
    occu_embed: np.array
        - occupation word embedding
        - shape: (# data, embed_size)
    """

    if embed_size != 300:
        fasttext.util.reduce_model(ft, embed_size)

    occu_embed = np.zeros((user_df.shape[0], embed_size))

    for i, occus in enumerate(user_df['occupation_titles']):
        # not nan
        if type(occus) == str:
            if ',' in occus:
                occus = occus.split(',')
            else:
                occus = [occus]

            word_vec = np.zeros((1, embed_size))
            for occu in occus:
                word_vec += ft.get_word_vector(occu)
        # give (0, ..., 0) to nan 
        else:
            word_vec = np.zeros((1, embed_size))
        occu_embed[i, :] = word_vec

    return occu_embed

def create_interest_embed(user_df, ft, embed_size=300):
    """
    Compute interest embedding.

    Parameters
    ----------
    user_df : pd.DataFrame
        - Should contains `interests` column
        - each data point in `interests` column looks like 'MainInterest_SubInterest'
    ft: fasttext word vector model
    embed_size: size of word embedding

    Returns
    -------
    main_int_embed: np.array
        - main interest word embedding
        - shape: (# data, embed_size)

    sub_int_embed: np.array
        - subinterest word embedding
        - shape: (# data, embed_size)
    """


    if embed_size != 300:
        fasttext.util.reduce_model(ft, embed_size)


    main_int_embed, sub_int_embed = \
            np.zeros((user_df.shape[0], embed_size)), \
                np.zeros((user_df.shape[0], embed_size))
    
    for i, interest in enumerate(user_df['interests']):
        if type(interest) == str:
            # user has serveral interests
            if ',' in interest:
                int_split = interest.split(',')
                main_int_set = set()
                for int_ in int_split:
                    main_int = int_.split('_')[0]
                    sub_int = int_.split('_')[1]
                    main_int_set.update([main_int])
                    sub_int_embed[i, :] += ft.get_word_vector(sub_int)

                # mulitple main interest only counts once
                for main_int in main_int_set:
                    main_int_embed[i, :] += ft.get_word_vector(main_int)
            
            # user has only one interest
            else:
                main_int = interest.split('_')[0]
                sub_int = interest.split('_')[1]

                main_int_embed[i, :] += ft.get_word_vector(main_int)
                sub_int_embed[i, :] += ft.get_word_vector(sub_int)

    return main_int_embed, sub_int_embed

def create_recreation_embed(user_df, ft, embed_size=300):
    """
    Compute recreation embedding.

    Parameters
    ----------
    user_df : pd.DataFrame
        - Should contains `creation_names` column
    ft: fasttext word vector model
    embed_size: size of word embedding

    Returns
    -------
    rec_embed: np.array
        - recreation word embedding
        - shape: (# data, embed_size)
    """
    if embed_size != 300:
        fasttext.util.reduce_model(ft, embed_size)

    rec_embed = np.zeros((user_df.shape[0], embed_size))

    for i, recs in enumerate(user_df['recreation_names']):
        # not nan
        if type(recs) == str:
            if ',' in recs:
                recs = recs.split(',')
            else:
                recs = [recs]

            word_vec = np.zeros((1, embed_size))
            for occu in recs:
                word_vec += ft.get_word_vector(occu)
        # give (0, ..., 0) to nan 
        else:
            word_vec = np.zeros((1, embed_size))
        rec_embed[i, :] = word_vec

    return rec_embed


def create_user_embed(user_df, ft_path, do_knn_impute=False, knn_neighbors=10):
    ft = load_ft_model(ft_path)

    # one hot
    gender_embed = create_gender_embbeding(user_df)

    # word vec
    main_int_embed, sub_int_embed = create_interest_embed(user_df, ft, embed_size=300)
    occu_embed = create_occupation_embed(user_df, ft, embed_size=300)
    rec_embed = create_recreation_embed(user_df, ft, embed_size=300)

    # (number of users, embed_size)
    user_embed = np.concatenate(
        [gender_embed, occu_embed, main_int_embed, sub_int_embed, rec_embed],
        axis=1
    )

    if do_knn_impute:
        user_embed = user_embed.astype('float')
        # 0 to nan
        user_embed[user_embed==0] = np.nan
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        imputer.fit_transform(user_embed)
    else:
        imputer = None

    return user_embed, imputer