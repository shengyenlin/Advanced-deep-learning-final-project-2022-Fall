#!bin/bash
wget https://www.dropbox.com/s/4c0u3pi82hl210z/als_course_score.dict.zip?dl=1 -O ./als_course_score.dict.zip
unzip ./als_course_score.dict.zip -d ./ensemble/data
rm -f ./als_course_score.dict.zip
