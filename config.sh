pip install -q kaggle
mkdir ~/.kaggle
echo '{"username":"$1","key":"$2"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
~/.local/bin/kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
apt install unzip
unzip jigsaw-toxic-comment-classification-challenge.zip
unzip train.csv.zip
unzip test.csv.zip
unzip test_labels.csv.zip
