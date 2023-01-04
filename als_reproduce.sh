#!bin/bash
cd './ALS'
python3 ALS.py --train_file "../hahow/data/train/train.csv" --val_file "../hahow/data/val/val_seen.csv" --test_file "../hahow/data/test/test_seen.csv" --pred_file "./pred.csv"
cd ..