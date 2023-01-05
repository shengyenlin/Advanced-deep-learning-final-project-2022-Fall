pushd './logistic_regression/'
# train course model
bash ./train_course.bash
# train topic model
bash ./train_topic.bash
popd