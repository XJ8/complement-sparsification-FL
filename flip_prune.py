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
import data_utils
import tensorflow_model_optimization as tfmot
import tensorflow as tf
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
import numpy as np

def PQSU(global_model,x,y,server_percent,server_epoch):
    pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(server_percent, 0),
    'block_size': (1, 1),
    'block_pooling_type': 'AVG'
    }
    model_for_pruning = prune_low_magnitude(global_model, **pruning_params)
    model_for_pruning.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    model_for_pruning.fit(x, y, epochs=server_epoch, callbacks=callbacks,verbose=0)
    new_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    # global_model.summary()
    # quantize_model = tfmot.quantization.keras.quantize_model
    # q_aware_model = quantize_model(global_model)
    # q_aware_model.summary()
    # q_aware_model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    # q_aware_model.fit(x, y, epochs=server_epoch, callbacks=callbacks,verbose=0)
    # converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # quantized_tflite_model = converter.convert()
    # quantized_tflite_model.summary()
    # return q_aware_model.get_weights()
    return new_model.get_weights()

def flip_prune(global_model, aggregated_weights, NO_FINE_TUNE, data_dir, EID, rnd, server_percent,server_epoch, lr):
    # aggregated_weights = parameters_to_weights(new_aggregated_parameters)
    if (rnd>1):
        if NO_FINE_TUNE:
            print(f"Load dataless pruned round-{rnd-1}-weights.npz as previous_weights")
            previous_weights = np.load(f"models/{EID}/dataless_pruned/round-{rnd-1}-weights.npy", allow_pickle=True)
        else:
            print(f"Load data pruned round-{rnd-1}-weights.npz as previous_weights")
            previous_weights = np.load(f"models/{EID}/data_pruned/round-{rnd-1}-weights.npy", allow_pickle=True)
        # print(previous_weights[0])
        for i in range(len(previous_weights)):
            if i%2==0:
            # if i in prune_layers_index:
                # aggregated_weights[i] = previous_weights[i]+lr*(aggregated_weights[i]-previous_weights[i])
                aggregated_weights[i] = previous_weights[i]+lr*aggregated_weights[i]
    np.save(f"models/{EID}/aggregated/round-{rnd}-weights", aggregated_weights)
    # print(aggregated_weights)
    # global_model=build_model()
    # global_model = femnist_model()
    global_model.set_weights(aggregated_weights)
    # global_model_copy = tf.keras.models.clone_model(model_for_pruning)
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(server_percent, 0),
        'block_size': (1, 1),
        'block_pooling_type': 'AVG'
    }
    model_for_pruning = prune_low_magnitude(global_model, **pruning_params)
    # model_for_pruning_copy = tf.keras.models.clone_model(model_for_pruning)
    model_for_pruning.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    # x_train, y_train= data_utils.get_all_test_data(data_dir)
    x_train, y_train= data_utils.get_aug_data(data_dir)
    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    # tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    #   prune_logger
    ]
    if NO_FINE_TUNE:
        print(f"Perform data pruning without using data pruned model.")
    model_for_pruning.fit(x_train, y_train, epochs=server_epoch, callbacks=callbacks,verbose=0)
    global_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    # layer_index = 0
    # for layer in global_model.layers:
    #     layer.set_weights(aggregated_weights[layer_index])
    #     layer_index+=1
    global_model_weights = global_model.get_weights()
    if (global_model_weights is not None):
        # Save aggregated_weights
        print(f"Saving round {rnd} data pruned global_model_weights...")
        np.save(f"models/{EID}/data_pruned/round-{rnd}-weights", global_model_weights)
    # print(global_model_weights)
    new_aggregated_parameters = weights_to_parameters(global_model_weights)
    return global_model_weights, aggregated_weights, new_aggregated_parameters

def dataless_prune(global_model_weights, aggregated_weights,EID,rnd):
    mask = []
    for i in range(len(global_model_weights)):
    # print(parameters[2].shape)
        if i%2==0:
        # if i in prune_layers_index:
            # zero_array = np.zeros(global_model_weights[i].shape)
            # one_array = np.ones(global_model_weights[i].shape)
            # one_array = np.ones(global_model_weights[i].shape)
            # np.copyto(layer_mask, one_array, where=(layer_mask!= 0))
            layer_mask = global_model_weights[i].copy()
            one_array = np.ones(global_model_weights[i].shape)
            np.copyto(layer_mask, one_array, where=(layer_mask!= 0))
            mask.append(layer_mask)
    # print(mask)
    for i in range(len(aggregated_weights)):
        if i%2==0:
        # if i in prune_layers_index:
            aggregated_weights[i] = aggregated_weights[i]*mask[int(i/2)]
    # print(aggregated_weights)
    print(f"Saving round {rnd} dataless pruned global_model_weights...")
    np.save(f"models/{EID}/dataless_pruned/round-{rnd}-weights", aggregated_weights)
    new_aggregated_parameters = weights_to_parameters(aggregated_weights)
    return new_aggregated_parameters
