# download cache
wget -O cache.zip https://www.dropbox.com/s/laj26bngcg80szu/cache.zip?dl=1
unzip cache.zip -d logistic_regression
rm -f cache.zip

# download dictionary
wget -O lg_dict.zip https://www.dropbox.com/s/tkrk34bjieyql0n/lg_dict.zip?dl=1
unzip ./lg_dict.zip -d ./ensemble/data
rm -f lg_dict.zip

# download fasttext
cd logistic_regression && \
    mkdir FastText && \
    cd FastText && \
    conda activate logistic_regression && \
    python -c "import fasttext.util; fasttext.util.download_model('zh', if_exists='ignore')"