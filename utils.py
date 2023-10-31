import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_score,recall_score,accuracy_score

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

def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

## recommendation list eval
def evaluation(usr_list, test_like, test_dislike, recommend_list):
    precision = []
    full_precision = []
    recall = []
    for usr in usr_list:
        rec = set(recommend_list[usr])
        like = set(test_like[usr])
        dislike = set(test_dislike[usr])
        
        TP = len( rec & like )
        FP = len( rec & dislike )
        
        p = TP / (TP + FP) if TP + FP > 0 else np.nan
        fp = TP / len(rec) if len(rec)> 0 else np.nan
        r = TP / len(like) if len(like) > 0 else np.nan

        recall.append(r)
        precision.append(p)
        full_precision.append(fp)

    print(f"precision: {np.nanmean(precision):.4f}, \
          full precision: {np.nanmean(full_precision):.4f}, \
          recall: {np.nanmean(recall):.4f}")

## usr like or not clf eval
def eval(y_true, y_pred):

    y_pred = torch.round(torch.sigmoid(y_pred)).detach().numpy()
    y_true = y_true.detach().numpy()

    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true,y_pred)
    return p, r, acc
