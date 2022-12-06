import os,pickle, gc, time,csv
import math
from tkinter import NO
import data_utils
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf
import numpy as np
from typing import Dict, Callable, Optional, Tuple, List
from flwr.common import (
    # EvaluateIns,
    # EvaluateRes,
    # FitIns,
    # FitRes,
    # MetricsAggregationFn,
    # Parameters,
    # Scalar,
    # Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_model_optimization as tfmot
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
from models import femnist_model
import flip_prune

data_dir='/data/yours/flower_femnist/'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3";
os.environ["CUDA_VISIBLE_DEVICES"]="-1";
# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# The experimetn ID to differ the directory for saving results.
EID = 'fe67' 
# True, Use complement sparsification.
FLIP_PRUNE = True
# False. Do not use it. This is to use some data for the server to fine tune an aggregated model. 
FINE_TUNE = False
# True. This is what we propose. Use complement sparsification without fine-tuning the pruned model on the server. 
NO_FINE_TUNE = True
# This is one of the baselines.
PQSU = False
server_lr = 1.5
opt=Adam(learning_rate=0.01)
train_ratio = 0.0025
# train_ratio = 0.1154
min_user = 10
eval_ratio = 1
server_percent_start = 0.5
server_percent = 0.5
sim_seed = 13
local_epoch = 5
batch = 64
# This is to use tensorflow API to simplfy the implementation. We don't actually fine-tune the pruned model on the server.
server_epoch = 1
n_round = 500
start = 0
continue_model = False
# continue_model = "models/fe32/dataless_pruned/round-500-weights.npy"

tf.random.set_seed(sim_seed)
prune_layers_index = [1,3,4,7,8]

NUM_CLIENTS = len(os.listdir(data_dir))-1
# NUM_CLIENTS = 20

# if not os.path.exists(EID):
#     os.makedirs(EID)

if not os.path.exists('models/%s/aggregated/'%EID):
    os.makedirs('models/%s/aggregated/'%EID)

if not os.path.exists('models/%s/data_pruned/'%EID):
    os.makedirs('models/%s/data_pruned/'%EID)

if not os.path.exists('models/%s/dataless_pruned/'%EID):
    os.makedirs('models/%s/dataless_pruned/'%EID)

if NO_FINE_TUNE:
    FLIP_PRUNE = True
    FINE_TUNE = False
    # server_epoch = 10

# is_first_round = True

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x,y, x_test,y_test) -> None:
        super().__init__()
        self.model = model
        # self.cid = cid
        # self.data_path = data_dir
        self.x_train=x
        self.y_train=y
        self.x_val=x_test
        self.y_val=y_test


    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # self.model.summary()
        # print(parameters)
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, batch_size=batch,epochs=local_epoch, verbose=0)
        client_weights = self.model.get_weights()
        # print(client_weights)
        # print(len(parameters))
        # for i in range(len(parameters)):
        #     print(parameters[i].shape)
        server_pruned = not np.all(parameters[0])
        if FLIP_PRUNE and server_pruned:
            # print("Flip pruning on client.")
            flipped_mask = []
            for i in range(len(parameters)):
            # print(parameters[2].shape)
                if i%2==0:
                    mask = parameters[i].copy()
                    one_array = np.ones(parameters[i].shape)
                    np.copyto(mask, one_array, where=(mask!= 0))
                    flipped_mask.append(1-mask)
            # print(flipped_mask)
            # print(len(flipped_mask))
            for i in range(len(client_weights)):
                if i%2==0:
                    # print(client_weights[i].shape)
                    # print(flipped_mask[int(i/2)].shape)
                    client_weights[i] = client_weights[i] * flipped_mask[int(i/2)]
            for i in range(len(client_weights)):
                if i%2==0:
                    file = open('percent/%s.csv'%EID, 'a+', newline ='')
                    with file:
                        write = csv.writer(file)
                        write.writerow([i,tf.size(client_weights[i]).numpy(),tf.math.count_nonzero(client_weights[i]).numpy()])
                # print(tf.size(client_weights[i]).numpy())
                # print(tf.math.count_nonzero(client_weights[i]).numpy())
        if PQSU:
            client_weights = flip_prune.PQSU(self.model,self.x_train, self.y_train,0.9,5)

        # # print(len(client_weights))
        # return self.model.get_weights(), len(self.x_train), {}
        # print(client_weights)
        # del model
        # gc.collect()
        # tf.keras.backend.clear_session()
        return client_weights, len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        # x_val, y_val = data_utils.get_test_data(self.data_dir, self.cid)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        # print(acc)
        return loss, len(self.x_val), {"accuracy": acc}

def client_fn(cid: str) -> fl.client.Client:
    # Load model
    # model = build_model()
    model = femnist_model()
    # model.summary()
    # lr = 0.00001
    # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

    with open('%s/%s/train.pickle'%(data_dir,cid), 'rb') as f:
        x_train, y_train = pickle.load(f)
    with open('%s/%s/test.pickle'%(data_dir,cid), 'rb') as f:
        x_test, y_test = pickle.load(f)
    return FlwrClient(model,x_train,y_train, x_test, y_test)

def get_eval_fn():
    """Return an evaluation function for server-side evaluation."""

    # Use the last 5k training examples as a validation set
    x_val, y_val = data_utils.get_all_test_data(data_dir)

    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        # model = build_model()
        model = femnist_model()
    # model.summary()
    # lr = 0.00001
        # model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        del model
        gc.collect()
        tf.keras.backend.clear_session()
        return loss, {"accuracy": accuracy}

    return evaluate

# full_weights = 0

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[fl.common.Weights]:
        print("Round %s"%rnd)
        global start_time
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        aggregated_parameters = super().aggregate_fit(rnd, results, failures)
        new_aggregated_parameters = aggregated_parameters[0]
        aggregated_weights = parameters_to_weights(new_aggregated_parameters)
        if continue_model and rnd==1:
            print(f"Load previous_weights and continue training.")
            aggregated_weights = np.load(continue_model, allow_pickle=True)
            new_aggregated_parameters = weights_to_parameters(aggregated_weights)
        if FINE_TUNE==False and FLIP_PRUNE==False and NO_FINE_TUNE==False:
            # aggregated_weights = parameters_to_weights(new_aggregated_parameters)
            print(f"FL saving round {rnd} global_model_weights...")
            np.save(f"models/{EID}/aggregated/round-{rnd}-weights", aggregated_weights)
        # print(parameters_to_weights(new_aggregated_parameters))
        # full_weights = parameters_to_weights(new_aggregated_parameters)
        if rnd <= start:
            np.save(f"models/{EID}/aggregated/round-{rnd}-weights", aggregated_weights)
            np.save(f"models/{EID}/data_pruned/round-{rnd}-weights", aggregated_weights)
            np.save(f"models/{EID}/dataless_pruned/round-{rnd}-weights", aggregated_weights)
        if FINE_TUNE:
            # global_model=build_model()
            global_model = femnist_model()
            global_model.set_weights(aggregated_weights)
            global_model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
            x_train, y_train= data_utils.get_aug_data(data_dir)
            global_model.fit(x_train, y_train, epochs=server_epoch)
            new_aggregated_parameters = weights_to_parameters(global_model.get_weights())
        if rnd > start:
            if FLIP_PRUNE:
                global_model=femnist_model()
                # global lr
                # lr = lr/rnd
                sp = server_percent_start + rnd*(server_percent-server_percent_start)/n_round
                global_model_weights, aggregated_weights, new_aggregated_parameters = flip_prune.flip_prune(global_model,aggregated_weights, NO_FINE_TUNE, data_dir, EID,rnd, sp,server_epoch, server_lr)
                if NO_FINE_TUNE:
                    new_aggregated_parameters = flip_prune.dataless_prune(global_model_weights,aggregated_weights,EID,rnd)
        return new_aggregated_parameters,aggregated_parameters[1]

def main() -> None:
    # Start Flower simulation
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    #     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 4},
        # client_resources={"num_gpus": 0.5,"num_cpus": 0.5},
        num_rounds=n_round,
        # strategy=fl.server.strategy.FedAvg(
        strategy= SaveModelStrategy(
            eval_fn=get_eval_fn(),
            fraction_fit=train_ratio,
            fraction_eval=eval_ratio,
            min_fit_clients=min_user,
            min_eval_clients=1,
            min_available_clients=NUM_CLIENTS,
        ),
    )

if __name__ == "__main__":
    print("%s starts."%EID)
    start_time = time.time()
    main()
    print("%s completes."%EID)
