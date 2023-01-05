#!bin/bash
wget https://www.dropbox.com/s/wx7h81e4xinn78l/als_course_score.dict.zip?dl=1 -O ./als_course_score.dict.zip
unzip ./als_course_score.dict.zip -d ./ensemble/data
rm -f ./als_course_score.dict.zip
