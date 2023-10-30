from tqdm import tqdm
from argparse import ArgumentParser
from sortedcontainers import SortedList
from collections import defaultdict
from sklearn.model_selection import train_test_split
from utils import dict_set, cal_similarity, mse, evaluation
import pandas as pd
import numpy as np


def get_like_dislike(test_dict):
    like_list, dislike_list = defaultdict(set), defaultdict(set) 
    for usr, ratings in tqdm(test_dict.items()):
        for item, rate in ratings.items():
            if rate >=4: like_list[usr].add(item)
            else: dislike_list[usr].add(item)
    return like_list, dislike_list 


def training(train_data, k=10):
    neighbors, devs, avgs = defaultdict(list),  defaultdict(float),  defaultdict(float)
    for key, rate_results in tqdm(train_data.items()):
        
        # avg
        avg = np.mean(list(rate_results.values()))
        avgs[key]=avg

        # devs
        tmp_dev = {item: ratings-avg for item, ratings in rate_results.items()}
        devs[key] = tmp_dev

        # neighbors & weights
        tmp_neighgors = SortedList()
        for key1, rate_results1 in train_data.items():
            if key == key1: continue

            sim = cal_similarity(rate_results, rate_results1)
            tmp_neighgors.add((-sim, key1))
            if len(tmp_neighgors) > k: del tmp_neighgors[-1]

        neighbors[key] = tmp_neighgors
        
    return neighbors, devs, avgs


def predict_rate(data_dict, neighbors, devs, avgs, limit=5):
    results = []
    targets = []
    for usr, value in data_dict.items():
        for _, rate in value.items():
            numerator = 0
            denominator = 0
            predict = avgs[usr]
            if len(neighbors[usr]) >= limit:
                for w, nei in neighbors[usr]:
                    try: 
                        w = -w
                        numerator += w * devs[usr][nei]
                        denominator += abs(w)
                    except KeyError: # user didn't watch movie that neighbot watch
                        pass
                if denominator != 0: 
                    predict += numerator / denominator
            
            predict  = min(5, predict)
            predict  = max(0.5, predict) # min rating is 0.5
            targets.append(rate)
            results.append(predict)
    return results, targets


def gen_recommend(data_set, neighbors, train_like):
    recommend = defaultdict(set)
    for usr, ratings in tqdm(data_set.items()):
        watched_list = set(ratings.keys())
        
        neighbor_watched_list = set()
        for _, u in neighbors[usr]: 
            neighbor_watched_list |= train_like[u] #add movie that similiar neighbor like

        recommend[usr] |= (neighbor_watched_list - watched_list)

    return recommend


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", dest="sample", help="use smaller data set or not", type=bool, default=True)
    args = parser.parse_args()
    
    ## load data 
    if args.sample:
        data = pd.read_csv('data/small_rating.csv')
    else: 
        data = pd.read_csv('data/procressed_rating.csv')
        
    print(f"sample:{args.sample}, data: {data.shape}")
    
    ## train / test split
    print("=== train, test split")
    train, test= train_test_split(data, test_size=0.2, random_state=1126)
    print(f"train:{train.shape}, test:{test.shape}")


    ## get user-item-rate dataset
    print("=== create user-item-rate data set")
    train_dict = dict_set(train, key='user_id', pair1='movie_id', rate='rating')
    test_dict = dict_set(test, key='user_id', pair1='movie_id', rate='rating')

    ## create like, dislike list
    train_like, _ = get_like_dislike(train_dict)
    test_like, test_dislike = get_like_dislike(test_dict)
   

    ## training: cal similarity(weight), avg, dev
    print("=== training")
    neighbors, devs, avgs = training(train_dict, k=20) # consdier 20 neighbors
    print(f"size: neighbors: {len(neighbors)}, devs: {len(devs)}, avgs:{len(avgs)}")
    
    ## predict
    print("=== predict rating")
    pre_train, tar_train = predict_rate(train_dict, neighbors, devs, avgs, limit=5) # if neighbor# < limit, use avg
    print('train mse:', mse(pre_train, tar_train))
    
    pre_test, tar_test = predict_rate(test_dict, neighbors, devs, avgs, limit=5)
    print('test mse:', mse(pre_test, tar_test))
    

    ## generate recommend movie list
    print("=== Generate recommend & cal precision, recall")
    recommend_list = gen_recommend(train_dict, neighbors, train_like)
    evaluation(set(test_dict.keys()), test_like, test_dislike, recommend_list)

