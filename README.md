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

## Collaborative Filtering by K-Nearest Neighber
Implement collaborative filtering(cf) recommendation by k-nearest neighber method.
```shell
# use -a flag for traing & predict for all data
python item_based_cf.py #item-based cf
python usr_based_cf.py  #user-based cf
```
#### result for 5000 users & 300 movies (train:80%, test:20%)
Calculate k = 20 nearest(most similiar) neighbor to do score estimatnio and recommendation. 
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
 
## Collaborative Filtering by Matrix Factorization
Implement one of matrix factorization method - ALS (Alternating Least Squares) to decomposes the user-item interaction matrix into user and item feature matrices and predict whether or not item would be like.
* like: rating 4~5
* dislike: rating 1~3
```shell
# use -a flag for traing & predict for all data
python als_algo.py -a
```
#### Result (train:80%, test:20%)

| Dataset | Loss | Accuracy | Precision | Recall |
|  ----  | ----  | ---- | ----  | ----  |
| trianing set | 0.57 | 0.73 | 0.74 | 0.83 | 
| test set  | 0.58 | 0.73 | 0.73 | 0.82 |

## Collaborative Filtering with Users & Items Features
Implement Logistic Regression, Poly2(Degree-2 Ploynomial Margin) and FM (Factorization Machines) to predict whether or not item would be like.
* like: rating 4~5
* dislike: rating 1~3
```shell
# -a: flag for traing & predict for all data
# -e: model training epochs(default:5)
# -m: model stype ("lr": logistic regression, "poly":poly2, "fm1": Traditional FM, "fm2": Embedding FM)
python lr_ploy_fm.py -a
```
#### Result (train:80%, test:20%)
* Logistic Regression

| Dataset | Loss | Accuracy | Precision | Recall |
|  ----  | ----  | ---- | ----  | ----  |
| trianing set | 0.61 | 0.69 | 0.69 | 0.99 | 
| test set  | 0.61 | 0.69 | 0.69 | 0.99 |

* Poly2

| Dataset | Loss | Accuracy | Precision | Recall |
|  ----  | ----  | ---- | ----  | ----  |
| trianing set | 0.60 | 0.70 | 0.70 | 0.96 | 
| test set  | 0.60 | 0.70 | 0.71 | 0.97 |

* Traditional FM: FM is a modified method of Poly2 by applying matrix multiplication to reduce time complexity.

| Dataset | Loss | Accuracy | Precision | Recall |
|  ----  | ----  | ---- | ----  | ----  |
| trianing set | 0.60 | 0.70 | 0.70 | 0.97 | 
| test set  | 0.60 | 0.70 | 0.71 | 0.96 |

* Embedding FM (embedding dim = 128): Use embedding to represent one-hot X(item & user feats.) * v(implicit weight vector for x) 

| Dataset | Loss | Accuracy | Precision | Recall |
|  ----  | ----  | ---- | ----  | ----  |
| trianing set | 0.62 | 0.69 | 0.69 | 1.00 | 
| test set  | 0.61 | 0.69 | 0.69 | 1.00 |
