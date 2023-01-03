# python3 logistic_regression/topic_prediction.py \
#     --data_dir ./hahow/data \
#     --model logistic_regression \
#     --mode 'test' \
#     --remove_nan \
#     --ft_path ./logistic_regression/FastText/cc.zh.300.bin \
#     --sbert_dir utils/sbert \
#     --use_sbert_interaction \
#     --cos_sim_interaction \
#     --normalize_feature \
#     --normalizer_path logistic_regression/best_models/log_reg_train_normalizer.pkl \
#     --seen_model_path logistic_regression/best_models/topic_seen_model.joblib \
#     --unseen_model_path logistic_regression/best_models/topic_unseen_model.joblib \
#     --out_dir prediction

    # --normalize_pred_prob
    # --impute_knn \
    # --neighbor_for_impute 10 \

python3 logistic_regression/course_prediction.py \
    --data_dir ./hahow/data \
    --model logistic_regression \
    --mode 'test' \
    --remove_nan \
    --ft_path ./logistic_regression/FastText/cc.zh.300.bin \
    --sbert_dir utils/sbert \
    --use_sbert_interaction \
    --cos_sim_interaction \
    --normalize_feature \
    --normalizer_path logistic_regression/best_models/log_reg_train_normalizer.pkl \
    --seen_model_path logistic_regression/best_models/course_seen_model.joblib \
    --unseen_model_path logistic_regression/best_models/course_unseen_model.joblib \
    --out_dir ensemble/data
    # --normalize_pred_prob
    # --impute_knn \
    # --neighbor_for_impute 10 \