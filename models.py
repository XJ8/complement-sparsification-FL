import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten,Reshape

model_seed = 13
init = 'he_uniform'

tf.random.set_seed(model_seed)

# sent_init = 'lecun_uniform'
sent_init = init

def build_model():
    model = Sequential(
    [Dense(32, activation="relu",input_shape=(768,), kernel_initializer=sent_init),
    Dense(3, activation="softmax")
    ])
    return model

def femnist_model():
    model = Sequential()
    model.add(Reshape((28,28,1),input_shape=(784,)))
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=init , input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=init))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=init))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer=init))
    model.add(Dense(62, activation='softmax', kernel_initializer=init))
    # compile model
#     opt = SGD(learning_rate=0.01, momentum=0.9)
#     model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
