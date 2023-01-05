#!bin/bash
wget https://www.dropbox.com/s/4c0u3pi82hl210z/ALS_course_score.dict.zip?dl=1 -O ./ALS_course_score.dict.zip
unzip ./ALS_course_score.dict.zip -d ./ensemble/data
rm -f ./ALS_course_score.dict.zip