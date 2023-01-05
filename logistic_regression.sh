python3 logistic_regression/topic_prediction.py \
    --data_dir ./hahow/data \
    --model logistic_regression \
    --mode 'test' \
    --remove_nan \
    --ft_path ./logistic_regression/FastText/cc.zh.300.bin \
    --sbert_dir utils/sbert \
    --use_sbert_interaction \
    --cos_sim_interaction \
    --normalize_feature \
    --cache_dir ./logistic_regression/cache \
    --ensemble_dir ./ensemble/data
    --pred_dir ./prediction/
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
    --cache_dir ./logistic_regression/cache \
    --ensemble_dir ./ensemble/data
    --pred_dir ./prediction/
    # --normalize_pred_prob
    # --impute_knn \
    # --neighbor_for_impute 10 \