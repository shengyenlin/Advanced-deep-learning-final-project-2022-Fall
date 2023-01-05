# download cache
wget -O cache.zip https://www.dropbox.com/s/laj26bngcg80szu/cache.zip?dl=1
unzip cache.zip -d logistic_regression

# download fasttext
cd logistic_regression && \
    mkdir FastText && \
    cd FastText && \
    python -c "import fasttext.util; fasttext.util.download_model('zh', if_exists='ignore')"