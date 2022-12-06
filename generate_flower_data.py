# import required module
import os
import pandas as pd
import random
import io
from sklearn.model_selection import train_test_split
import pickle
from leafdata_utils import read_data
import numpy as np

out_path = '/data/yours/flower_femnist'
split = 0.2

dataset = 'femnist'
eval_set = 'test'
train_data_dir = os.path.join('/data/yours/leaf-master', 'data', dataset, 'data', 'train')
test_data_dir = os.path.join('/data/yours/leaf-master', 'data', dataset, 'data', eval_set)

users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

if not os.path.exists(out_path):
    os.makedirs(out_path) 
 
# iterate over files in
# that directory
index = 0
test_x = []
test_y = []
for user in users:
    # f = os.path.join(directory, filename)
    # # checking if it is a file
    # if os.path.isfile(f):
    #     df =  pd.read_csv( f, header= None)
    #     labels = df.iloc[:,[0]]
    #     features = df.iloc[:,1:]
    #     train_features, test_features, train_labels, test_labels = train_test_split(features, labels,test_size=split, random_state=42)
    if not os.path.exists('%s/%s'%(out_path,index)):
        os.makedirs('%s/%s'%(out_path,index)) 
    with open('%s/%s/train.pickle'%(out_path,index), 'wb') as f:
        pickle.dump([train_data[user]['x'], train_data[user]['y']], f)
    user_test_x = test_data[user]['x']
    user_test_y = test_data[user]['y']
#     train_x.append(user_train_x)
#     train_y.append(user_train_y)
    test_x.append(user_test_x)
    test_y.append(user_test_y)
    with open('%s/%s/test.pickle'%(out_path,index), 'wb') as f:
        pickle.dump([user_test_x, user_test_y], f)
    index+=1

test_x_arr = np.concatenate(test_x)
test_y_arr = np.concatenate(test_y)
with open('%s/test.pickle'%(out_path), 'wb') as f:
    pickle.dump([test_x_arr,test_y_arr], f)