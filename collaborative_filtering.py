from tqdm import tqdm
from argparse import ArgumentParser
from sortedcontainers import SortedList
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import collections

def dict_set(df, key, pair1, rate):
    result = defaultdict(dict)
    for _, row in tqdm(df.iterrows()):
        result[row[key]][row[pair1]] = float(row[rate])
    return result

def cal_similarity(r1, r2):
    common = set(r1.keys()) & set(r2.keys())
    m1, m2 = np.mean(list(r1.values())), np.mean(list(r2.values()))
    arr1 = [v-m1 for k, v in r1.items() if k in common]
    arr2 = [v-m2 for k, v in r2.items() if k in common]
    return (np.dot(arr1,arr2)) / (np.linalg.norm(arr1)*np.linalg.norm(arr2))


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
    for key, value in data_dict.items():
        for item, rate in value.items():
            numerator = 0
            denominator = 0
            predict = avgs[key]
            if len(neighbors[key]) >= limit:
                for w, nei in neighbors[key]:
                    try: 
                        w = -w
                        numerator += w * devs[key][nei]
                        denominator += abs(w)
                    except KeyError: # user didn't watch movie that neighbot watch
                        pass
                if denominator != 0: predict += numerator / denominator
            
            predict  = min(5, predict)
            predict  = max(0.5, predict) # min rating is 0.5
            targets.append(rate)
            results.append(predict)
    return results, targets

def get_like_dislike_usrbased(test_dict):
    like_list, dislike_list = defaultdict(set), defaultdict(set) 
    for usr, ratings in tqdm(test_dict.items()):
        for item, rate in ratings.items():
            if rate >=4: like_list[usr].add(item)
            else: dislike_list[usr].add(item)
    return like_list, dislike_list 

def get_like_dislike_itembased(test_dict):
    like_list, dislike_list = defaultdict(set), defaultdict(set) 
    for item, ratings in tqdm(test_dict.items()):
        for usr, rate in ratings.items():
            if rate >=4: like_list[usr].add(item)
            else: dislike_list[usr].add(item)
    return like_list, dislike_list 

def gen_recommend_usrbased(data_set, neighbors):
    recommend = defaultdict(set)
    for usr, ratings in tqdm(data_set.items()):
        watched_list = set(ratings.keys())
        neighbor_watched_list = set()
        for _, u in neighbors[usr]: 
            neighbor_watched_list |= set(data_set[u].keys())
        
        recommend[usr] |= (neighbor_watched_list - watched_list)

    return recommend

def get_item_based_watched_list(data_set):
    watch_list = defaultdict(set)
    for item, ratings in data_set.items():
        for usr, _ in tqdm(ratings.items()):
            watch_list[usr].add(item)
        
    return watch_list

def gen_recommend_itembased(data_set, neighbors):
    recommend = defaultdict(set)
    watch_dict = get_item_based_watched_list(data_set)
    
    for usr, watched_list in tqdm(watch_dict.items()):
        neighbor_list = set()
        for item in watched_list:
            neighbor_list |= set(m for _, m in neighbors[item])
        recommend[usr] |= (neighbor_list - watched_list)
    return recommend

def evaluation(usr_list, test_like, test_dislike, recommend_list):
    precision = []
    recall = []
    for usr in usr_list:
        rec = set(recommend_list[usr])
        like = set(test_like[usr])
        dislike = set(test_dislike[usr])
        
        TP = len( rec & like )
        FP = len( rec & dislike )
        
        p = TP / (TP + FP) if TP + FP > 0 else np.nan
        r = TP / len(like) if len(like) > 0 else np.nan

        recall.append(p)
        precision.append(r)
    print(f"precision: {np.nanmean(precision):.4f}, recall: {np.nanmean(recall):.4f}")

def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", dest="sample", help="use smaller data set or not", type=bool, default=True)
    parser.add_argument("-t", dest="task", help="user-based:0, item-based:1", type=int, default=0)
    args = parser.parse_args()
    
    ## load data
    if args.sample:
        data = pd.read_csv('data/small_rating.csv')
    else: 
        data = pd.read_csv('data/new_rating.csv')
    print(f"sample:{args.sample}, data: {data.shape}")
    
    ## train / test split
    print("=== train, test split")
    train, test= train_test_split(data, test_size=0.2, random_state=1126)
    print(f"train:{train.shape}, test:{test.shape}")

    ## get user-item-rate dataset
    print("=== create user-item-rate data set")
    if args.task ==0:
        print("==== user based")
        train_dict = dict_set(train, key='userId', pair1='movieId', rate='rating')
        test_dict = dict_set(test, key='userId', pair1='movieId', rate='rating')
    else:
        print("==== item based")
        train_dict = dict_set(train, key='movieId', pair1='userId', rate='rating')
        test_dict = dict_set(test, key='movieId', pair1='userId', rate='rating')

    ## training: cal similarity(weight), avg, dev
    print("=== training")
    neighbors, devs, avgs = training(train_dict)
    print(f"size: neighbors: {len(neighbors)}, devs: {len(devs)}, avgs:{len(avgs)}")
    
    ## predict
    print("=== predict rating")
    pre_train, tar_train = predict_rate(train_dict, neighbors, devs, avgs, limit=5)
    print('train mse:', mse(pre_train, tar_train))
    
    pre_test, tar_test = predict_rate(test_dict, neighbors, devs, avgs, limit=5)
    print('test mse:', mse(pre_test, tar_test))
    

    ## generate recommend movie list
    print("=== Generate recommend & cal precision, recall")
    if args.task ==0: # usr based
        test_like, test_dislike = get_like_dislike_usrbased(test_dict)
        recommend_list = gen_recommend_usrbased(train_dict, neighbors)
        evaluation(set(test_dict.keys()), test_like, test_dislike, recommend_list)
    else: #item based
        test_like, test_dislike,  = get_like_dislike_itembased(test_dict)
        recommend_list = gen_recommend_itembased(train_dict, neighbors)
        usr_list = set(test_dislike.keys() | test_like.keys())
        evaluation(usr_list, test_like, test_dislike, recommend_list)
