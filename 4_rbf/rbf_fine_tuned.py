import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.datasets import boston_housing
from keras import layers
from keras.initializers import Initializer
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold


# Parameters
LR = 0.001
EPOCHS = 100
TESTING_PERCENTAGE = 0.25
VALIDATION_PERCENTAGE = 0.2
BATCH_SIZE = 16
KK = 5
NH1 = [0.05, 0.15, 0.3, 0.5]
NH2 = [32, 64, 128, 256]
DROPOUT = [0.2, 0.35, 0.5]


class RBFLayer(layers.Layer):
    def __init__(self, output_dim, initializer, betas=1.0, **kwargs):
        self.betas = betas
        self.output_dim = output_dim
        self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=False)
        d = [np.linalg.norm(i - j) for i in self.centers for j in self.centers]
        sigma = max(d) / np.sqrt(2 * self.output_dim)
        self.betas = np.ones(self.output_dim) / (2 * (sigma ** 2))
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        c = tf.expand_dims(self.centers, -1)
        h = tf.transpose(c - tf.transpose(inputs))
        return tf.exp(-self.betas * tf.math.reduce_sum(h ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class InitCentersKMeans(Initializer):
    def __init__(self, X, max_iter=100):
        self.X = X
        self.max_iter = max_iter
        super().__init__()

    def __call__(self, shape, dtype=None, *args):
        assert shape[1] == self.X.shape[1]
        n_centers = shape[0]
        km = KMeans(n_clusters=n_centers, max_iter=self.max_iter, verbose=0)
        km.fit(self.X)
        return km.cluster_centers_


def mse(true, pred):
    return K.mean(K.square(pred - true))


def rmse(true, pred):
    return K.sqrt(K.mean(K.square(pred - true)))


def r2(true, pred):
    return 1 - K.sum(K.square(true - pred)) / (K.sum(K.square(true - K.mean(true))) + K.epsilon())


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = boston_housing.load_data(test_split=TESTING_PERCENTAGE)
    train_x = (train_x - np.mean(train_x, axis=0)) / np.std(train_x, axis=0)
    test_x = (test_x - np.mean(test_x, axis=0)) / np.std(test_x, axis=0)
    train_x, val_x = tf.split(train_x, [train_x.shape[0] - int(train_x.shape[0] * VALIDATION_PERCENTAGE),
                                        int(train_x.shape[0] * VALIDATION_PERCENTAGE)], 0)
    train_y, val_y = tf.split(train_y, [train_y.shape[0] - int(train_y.shape[0] * VALIDATION_PERCENTAGE),
                                        int(train_y.shape[0] * VALIDATION_PERCENTAGE)], 0)
    x = tf.concat([train_x, val_x], 0).numpy()
    y = tf.concat([train_y, val_y], 0).numpy()

    neurons = [{
        'def': nh1 * train_y.shape[0],
        'nh1': f'%d%%' % int(nh1 * 100),
        'nh2': nh2,
        'dropout': p
    } for nh1 in NH1 for nh2 in NH2 for p in DROPOUT]

    optimal_loss = float('inf')
    optimal_neuron = None
    for neuron in neurons:
        rmses = []
        kf = KFold(KK, shuffle=True, random_state=1)
        for training, validation in kf.split(x):
            model = Sequential()
            model.add(RBFLayer(int(neuron['def']), initializer=InitCentersKMeans(x[training]), input_shape=(x[training].shape[1],)))
            model.add(Dense(neuron['nh2']))
            model.add(Dropout(neuron['dropout']))
            model.add(Dense(1))

            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LR), loss=mse, metrics=[rmse, r2])
            history = model.fit(x[training], y[training], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x[validation], y[validation]))
            rmses.append(min(history.history['rmse']))

        if sum(rmses) / len(rmses) < optimal_loss:
            optimal_loss = sum(rmses) / len(rmses)
            optimal_neuron = neuron

    # Optimal
    print(f'\nOptimal neuron is %s (nh2 = %d, dropout = %f)\n' % (optimal_neuron['nh1'], optimal_neuron['nh2'], optimal_neuron['dropout']))
    model = Sequential()
    model.add(RBFLayer(int(optimal_neuron['def']), initializer=InitCentersKMeans(train_x), input_shape=(train_x.shape[1],)))
    model.add(Dense(optimal_neuron['nh2']))
    model.add(Dropout(optimal_neuron['dropout']))
    model.add(Dense(1))

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LR), loss=mse, metrics=[rmse, r2])
    history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_x, val_y))
    score = model.evaluate(test_x, test_y)

    trn_loss = [history.history['loss'][i] for i in range(EPOCHS)]
    val_loss = [history.history['val_loss'][i] for i in range(EPOCHS)]
    trn_rmse = history.history['rmse']
    val_rmse = history.history['val_rmse']
    trn_r2 = history.history['r2']
    val_r2 = history.history['val_r2']

    plt.clf()
    plt.plot(trn_loss)
    plt.plot(val_loss)
    plt.title(f'Loss (Optimal, lr = %.3f)' % LR)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    plt.savefig('RBF_optimal_loss.png')

    plt.clf()
    plt.plot(trn_r2)
    plt.plot(val_r2)
    plt.title(f'R2 (Optimal, lr = %.3f)' % LR)
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(['Training R2', 'Validation R2'], loc='lower right')
    plt.savefig('RBF_optimal_R2.png')

    plt.clf()
    plt.plot(trn_rmse)
    plt.plot(val_rmse)
    plt.title(f'RMSE (Optimal, lr = %.3f)' % LR)
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend(['Training RMSE', 'Validation RMSE'], loc='upper right')
    plt.savefig('RBF_optimal_RMSE.png')

    print(f'\nStats for optimal neuron: MSE = %f, RMSE = %f, R2 = %f' % (score[0], score[1], score[2]))
