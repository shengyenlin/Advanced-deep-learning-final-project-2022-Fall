python3 topic_prediction.py \
    --model logistic_regression \
    --mode 'train' \
    --remove_nan \
    --use_sbert_interaction \
    --cos_sim_interaction \
    --normalize_feature;
    # --normalize_pred_prob
    # --impute_knn \
    # --neighbor_for_impute 10 \