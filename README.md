# recsys_algo
Implement different kinds of recommender system algorithm

## Dataset: [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/)
* download dataset and unzip to `data/`
* data preprocess: create smaller dataset (default 1000 users & 200 movies)
```shell
python data_preprocess.py 

## use -n, -m to change sampling users#(-n) or movies#(-m)
python data_preprocess.py -n 5000 -m 300
```

## Collaborative Filtering
Implement collaborative filtering recommendation method and evaluate rating estimation(MSE) and recommendation(precision, recall)
* training and run collaborative filtering recommendation
```shell
# use -a flag for traing & predict for all data
python item_based_cf.py #item-based cf
python usr_based_cf.py  #user-based cf
```

#### result for 5000 users & 300 movies (train:80%, test:20%, k=20)
* User-based cf

| Dataset | MSE | Precision| Recall |
|  ----  | ----  | ----  | ----  | 
| trianing set | 1.11| - | - |
| test set  | 1.14 | 0.72 |0.73 |

* Item-based cf

| Dataset | MSE | Precision| Recall |
|  ----  | ----  | ----  | ----  |
| trianing set | 1.28 | - | - | 
| test set  | 1.28 | 0.69 | 0.79 |
 