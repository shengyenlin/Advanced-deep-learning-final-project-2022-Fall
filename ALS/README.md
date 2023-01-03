# ALS

Put the mapping dictionary in ../utils/remap/

## How to reproduce 
**bash run_ALS.sh $1 $2 $3 $4**  
$1: path to the train csv file (e.g., ./hahow/data/train/train.csv)  
$2: path to the val csv file (e.g., ./hahow/data/val/val_seen.csv)   
$3: path to the test csv file (e.g., ./hahow/data/test/test_seen.csv)   
$4: path to the predcit csv file (e.g., ./pred.csv)   

## How to generate dictionary
**bash run_ALS_dict.sh $1 $2**
$1: Path to train data. (e.g., ./hahow/data/train/train.csv)  
$2: Path to generate dictionary (e.g., ensemble/data/als_course_score.dict.pkl)  
