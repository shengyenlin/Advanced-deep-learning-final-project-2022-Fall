pushd './LightGCN-PyTorch/code'
# For init sbert embeds
./download_sbert.sh
# Train
./run_seen_course_train.sh; ./run_unseen_course_train_sbert.sh; ./run_seen_group_train.sh; ./run_unseen_group_train_sbert.sh
# For enssemble
./run_hahow_all_assemble_score.sh
# Predict
./run_hahow_pred.sh
popd