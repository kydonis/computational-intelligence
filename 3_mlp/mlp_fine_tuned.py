import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils
from tensorflow.keras.regularizers import l2
from keras.initializers import HeNormal
from keras.callbacks import EarlyStopping
from keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

EPOCHS = 1000
KK = 5
NH1 = [64, 128]
NH2 = [256, 512]


def precision(true, pred):
    true_positives = K.sum(K.round(K.clip(true * pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(true, pred):
    true_positives = K.sum(K.round(K.clip(true * pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(true, pred):
    prec = precision(true, pred)
    rec = recall(true, pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(hp.Choice('units1', [64, 128]),
                                 kernel_regularizer=l2(hp.Choice('reg_value1', [0.1, 0.001, 0.000001])),
                                 activation='relu'))
    model.add(keras.layers.Dense(hp.Choice('units2', [256, 512]),
                                 kernel_regularizer=l2(hp.Choice('reg_value2', [0.1, 0.001, 0.000001])),
                                 activation='relu'))

    model.add(Dense(10, activation='softmax',
                    kernel_regularizer=l2(hp.Choice('reg_value3', [0.1, 0.001, 0.000001])),
                    kernel_initializer=HeNormal()))

    hp_learning_rate = hp.Choice('Learning_rate', values=[0.1, 0.01, 0.001])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[f1, precision, recall, tf.keras.metrics.CategoricalAccuracy()])
    return model


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_y = np_utils.to_categorical(train_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    train_x, valx = tf.split(train_x, [int(48000), int(12000)], 0)
    train_y, valy = tf.split(train_y, [int(48000), int(12000)], 0)
    x = tf.concat([train_x, valx], 0).numpy()
    y = tf.concat([train_y, valy], 0).numpy()

    tf.random.set_seed(89)
    tuner = kt.Hyperband(model_builder, objective=kt.Objective('f1', direction='max'), max_epochs=EPOCHS, hyperband_iterations=1, factor=4, overwrite=True)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='f1', patience=200, mode='max')
    tuner.search(train_x, train_y, epochs=EPOCHS, batch_size=1024, callbacks=[stop_early], validation_data=(valx, valy))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps)

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=256, validation_data=(valx, valy), callbacks=[EarlyStopping(monitor='val_f1', patience=200, mode='max')])
    trn_acc = [history.history['categorical_accuracy'][i] * 100 for i in range(len(history.history['categorical_accuracy']))]
    val_acc = [history.history['val_categorical_accuracy'][i] * 100 for i in range(len(history.history['val_categorical_accuracy']))]

    trn_loss = history.history['loss']
    val_loss = history.history['val_loss']

    trn_f1_m = history.history['f1']
    val_f1_m = history.history['val_f1']

    plt.clf()
    plt.plot(trn_f1_m)
    plt.plot(val_f1_m)
    plt.title('F1 measure ')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Measure')
    plt.legend(['Training F1 measure', 'Validation F1 measure'], loc='lower right')
    plt.savefig('F1_measure_Optimal.png')

    plt.clf()
    plt.plot(trn_loss)
    plt.plot(val_loss)
    plt.title('Losses ')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Losses', 'Validation Losses'], loc='lower right')
    plt.savefig('Losses_Optimal.png')

    plt.clf()
    plt.plot(trn_acc)
    plt.plot(val_acc)
    plt.title('Accuracy ')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    plt.savefig('Accuracy_Optimal.png')

    predict = model.predict(test_x)
    con_matrix = confusion_matrix(test_y.argmax(axis=1), predict.argmax(axis=1))
    print(con_matrix)
    print(accuracy_score(test_y.argmax(axis=1), predict.argmax(axis=1)))
    print(precision_score(test_y.argmax(axis=1), predict.argmax(axis=1), average='macro'))
    print(recall_score(test_y.argmax(axis=1), predict.argmax(axis=1), average='macro'))
    print(f1_score(test_y.argmax(axis=1), predict.argmax(axis=1), average='macro'))
