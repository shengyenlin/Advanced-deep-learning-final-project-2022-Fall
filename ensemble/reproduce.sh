# seen course
python main.py \
    --task 'course' \
    --seen \
    --save_path '../prediction/' \
    --lgn_weight 0.025 \
    --als_weight 0.95 \
    --lg_weight 0.025


# unseen course
python main.py \
    --task 'course' \
    --save_path '../prediction/' \
    --lgn_weight 0 \
    --als_weight 0.5 \
    --lg_weight 0.5


# seen topic
python main.py \
    --task 'topic' \
    --seen \
    --save_path '../prediction/' \
    --lgn_weight 0.4 \
    --lg_weight 0.6
