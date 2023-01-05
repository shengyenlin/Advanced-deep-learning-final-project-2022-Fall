# download cache
wget -O cache.zip https://www.dropbox.com/s/laj26bngcg80szu/cache.zip?dl=1
unzip cache.zip -d logistic_regression
rm -f cache.zip

# download dictionary
wget -O lg_group.zip https://www.dropbox.com/s/xi2xmn0zxvlyaq5/lg_group_score.dict.pkl.zip?dl=1
wget -O lg_course.zip https://www.dropbox.com/s/ljkchlsxyoi0xe2/lg_course_score.dict.pkl.zip?dl=1
unzip lg_group.zip -d ./ensemble/data
unzip lg_course.zip -d ./ensemble/data
rm -f lg_group.zip
rm -f lg_course.zip

# download fasttext
cd logistic_regression && \
    mkdir FastText && \
    cd FastText && \
    conda activate logistic_regression && \
    python -c "import fasttext.util; fasttext.util.download_model('zh', if_exists='ignore')"