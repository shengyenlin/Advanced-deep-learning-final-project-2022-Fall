#!bin/bash
python3 ALS.py --train_file "${1}" --val_file "${2}" --test_file "${3}" --pred_file "${4}"
