import pickle
def get_train_data(path, cid):
    with open('%s/%s/train.pickle'%(path,cid), 'rb') as f:
        X_train, y_train = pickle.load(f)    
    return X_train,y_train

def get_test_data(path, cid):
    with open('%s/%s/test.pickle'%(path,cid), 'rb') as f:
        X_test, y_test = pickle.load(f)    
    return X_test,y_test

def get_aug_data(path):
    with open('%s/aug.pickle'%(path), 'rb') as f:
        X_test, y_test = pickle.load(f)    
    return X_test,y_test

def get_all_test_data(path):
    with open('%s/test.pickle'%(path), 'rb') as f:
        X_test, y_test = pickle.load(f)    
    return X_test,y_test