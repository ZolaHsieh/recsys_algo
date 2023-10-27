# recsys_algo
Recommender system algorithm

## Dataset: [MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset/)
download dataset and unzip to `data/`

## Collaborative Filtering
Implement collaborative filtering recommendation method and evaluate rating estimation(MSE) and recommendation(precision, recall)
#### data: `rating.csv`
#### command
* data preprocess: reindex and create smaller dataset(default 1000 users & 200 movies)
```shell
python data_preprocess.py

## to change users#(-n) or movies#(-m)
python data_preprocess.py -n 5000 -m 300
```
* training and run collaborative filtering recommendation
```shell
python item_based_cf.py #item-based cf
python usr_based_cf.py  #user-based cf
```

#### result for 5000 users & 300 movies (train:80%, test:20%)
* User-based cf

| Dataset | MSE | Precision| Full Precision | Recall |
|  ----  | ----  | ----  | ----  | ---- |
| trianing set | 1.11| - | - | - |
| test set  | 1.12 | 0.56 | 0.19 | 0.87 |

* Item-based cf

| Dataset | MSE | Precision| Full Precision | Recall |
|  ----  | ----  | ----  | ----  | ---- |
| trianing set | 0.93 | - | - | - |
| test set  | 0.94 | 0.52 | 0.17  | 0.98 |
 