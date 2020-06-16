# toxic_comment_classification_274P

A BERT-based model that achieves an average AUC of 0.959 on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview) from Kaggle.

To download data, run:

`sudo setup.sh <kaggle username> <kaggle token>`
  
This will install the Kaggle API and the unzip utility and then create the kaggle.json authentication token file. Then, it downloads the dataset from Kaggle and unzips the CSV files.
