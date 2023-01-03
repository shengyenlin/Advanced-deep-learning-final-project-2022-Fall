# seen course
python main.py \
    --model 'mf' \
    --task 'train' \
    --eval 'seen'


# unseen course
python main.py \
    --model 'mf' \
    --task 'train_group' \
    --eval 'unseen'


# seen topic
python main.py \
    --model 'mf' \
    --task 'train_group' \
    --eval 'seen'


# unseen topic
python main.py \
    --model 'mf' \
    --task 'train_group' \
    --eval 'unseen'