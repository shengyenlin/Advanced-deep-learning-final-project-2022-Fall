python3 course_prediction.py \
    --data_dir ../hahow/data \
    --model logistic_regression \
    --mode 'train' \
    --remove_nan \
    --ft_path FastText/cc.zh.300.bin \
    --use_sbert_interaction \
    --cos_sim_interaction \
    --normalize_feature;
    # --normalize_pred_prob
    # --impute_knn \
    # --neighbor_for_impute 10 \