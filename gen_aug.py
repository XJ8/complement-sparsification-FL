# import required module
import numpy as np
import random
import data_utils
import pickle

in_path = '/data/yours/flower_femnist'
split = 0.05
n_user = 3597
random.seed(13)

users = random.sample(range(0, n_user), int(n_user*split))
 
# iterate over files in
# that directory
# index = 0
test_x = []
test_y = []
for user in users:
    x,y= data_utils.get_train_data(in_path,user)
    test_x.append(x)
    test_y.append(y)

test_x_arr = np.concatenate(test_x)
test_y_arr = np.concatenate(test_y)
with open('%s/aug.pickle'%(in_path), 'wb') as f:
    pickle.dump([test_x_arr,test_y_arr], f)