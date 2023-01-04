#!bin/bash
cd "./ALS"
python3 ALS_dict.py --data_path "../hahow/data/train/train.csv" --dict_path "../ensemble/data/als_course_score.dict.pkl"
cd ..