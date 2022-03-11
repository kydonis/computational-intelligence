import tensorflow as tf
import matplotlib.pyplot as plt
import time
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.regularizers import l1, l2
from keras.initializers import RandomNormal

# Parameters
EPOCHS = 100
VALIDATION_PERCENTAGE = 0.2
BATCH_SIZE = [1, 256, 48000]
RHO = [0.01, 0.99]
REGULAR = [l2(0.1), l2(0.01), l2(0.001)]
REGULAR_L1 = [l1(0.01)]
RC = [0.1, 0.01, 0.001, 0.01]
RCN = [2, 2, 2, 1]


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_y = np_utils.to_categorical(train_y, 10)
    test_y = np_utils.to_categorical(test_y, 10)
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    train_x, val_x = tf.split(train_x, [train_x.shape[0] - int(train_x.shape[0] * VALIDATION_PERCENTAGE),
                                        int(train_x.shape[0] * VALIDATION_PERCENTAGE)], 0)
    train_y, val_y = tf.split(train_y, [train_y.shape[0] - int(train_y.shape[0] * VALIDATION_PERCENTAGE),
                                        int(train_y.shape[0] * VALIDATION_PERCENTAGE)], 0)

    # 1. Learning with different batches
    tf.random.set_seed(89)
    for bs in BATCH_SIZE:
        print('For batch size = %d' % bs)
        start = time.time()
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])
        history = model.fit(train_x, train_y, batch_size=bs, epochs=EPOCHS, validation_data=(val_x, val_y), verbose=2)
        results = model.evaluate(test_x, test_y)

        trn_acc = [history.history['categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
        val_acc = [history.history['val_categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
        trn_los = history.history['loss']
        val_los = history.history['val_loss']

        print(trn_acc[-1])
        print(trn_los[-1])
        print(val_acc[-1])
        print(val_los[-1])
        print(results)

        plt.clf()
        plt.plot(trn_acc)
        plt.plot(val_acc)
        plt.title(f'Accuracy with batch = %d' % bs)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
        plt.savefig(f'Accuracy with batch = %d.png' % bs)

        plt.clf()
        plt.plot(trn_los)
        plt.plot(val_los)
        ax = plt.gca()
        ax.set_ylim([0, 3])
        plt.title(f'Loss with batch = %d' % bs)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Training Loss', 'Validation Accuracy'], loc='upper right')
        plt.savefig('Loss with batch = %d.png' % bs)

        print(f'Time for batch = %d is %f sec' % (bs, (time.time() - start)))

    # 2. RMSProp optimizer
    tf.random.set_seed(89)
    for r in RHO:
        print(f'For rho = %.2f' % r)
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=r),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        history = model.fit(train_x, train_y, batch_size=256, epochs=EPOCHS, validation_data=(val_x, val_y), verbose=0)

        results = model.evaluate(test_x, test_y)

        trn_acc = [history.history['categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
        val_acc = [history.history['val_categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
        trn_los = history.history['loss']
        val_los = history.history['val_loss']

        print(trn_acc[-1])
        print(trn_los[-1])
        print(val_acc[-1])
        print(val_los[-1])
        print(results)

        plt.clf()
        plt.plot(trn_acc)
        plt.plot(val_acc)
        plt.title(f'Accuracy with rho= %.2f' % r)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
        plt.savefig('Accuracy with rho= %.2f.png' % r)

        plt.clf()
        plt.plot(trn_los)
        plt.plot(val_los)
        ax = plt.gca()
        ax.set_ylim([0, 0.6])
        plt.title(f'Losses with rho= %.2f' % r)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
        plt.savefig(f'Losses with rho= %.2f.png' % r)

    # 3. SGD optimizer with weights from normal distribition with m=10
    tf.random.set_seed(89)
    print('At this moment regularization choice is lr=0.01 with W = 10')
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=10.0 / 255.0)))
    model.add(Dense(256, activation='relu', kernel_initializer=RandomNormal(mean=10.0 / 255.0)))
    model.add(Dense(10, activation='softmax', kernel_initializer=RandomNormal(mean=10.0 / 255.0)))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    history = model.fit(train_x, train_y, batch_size=256, epochs=EPOCHS, validation_data=(val_x, val_y), verbose=0)

    results = model.evaluate(test_x, test_y)

    trn_acc = [history.history['categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
    val_acc = [history.history['val_categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
    trn_los = history.history['loss']
    val_los = history.history['val_loss']

    print(trn_acc[-1])
    print(trn_los[-1])
    print(val_acc[-1])
    print(val_los[-1])
    print(results)

    plt.clf()
    plt.plot(trn_acc)
    plt.plot(val_acc)
    plt.title('(SGD) Accuracy with lr=0.01 with W = 10')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    plt.savefig('(SGD) Accuracy with lr=0.01 with W = 10 .png')

    plt.clf()
    plt.plot(trn_los)
    plt.plot(val_los)
    plt.title('(SGD) Losses with lr=0.01 with W = 10')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
    plt.savefig('(SGD) Losses with lr=0.01 with W = 10 .png')

    # 4. SGD with aε{0.1, 0.01, 0.001}
    tf.random.set_seed(89)
    for reg2 in range(0, len(REGULAR)):
        print('For regularization choice = l%d and a = %.3f' % (RCN[reg2], RC[reg2]))
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=10.0 / 255.0),
                        kernel_regularizer=REGULAR[reg2]))
        model.add(Dense(256, activation='relu', kernel_initializer=RandomNormal(mean=10.0 / 255.0),
                        kernel_regularizer=REGULAR[reg2]))
        model.add(Dense(10, activation='softmax', kernel_initializer=RandomNormal(mean=10.0 / 255.0),
                        kernel_regularizer=REGULAR[reg2]))
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.CategoricalAccuracy()])
        history = model.fit(train_x, train_y, batch_size=256, epochs=EPOCHS, validation_data=(val_x, val_y), verbose=0)
        results = model.evaluate(test_x, test_y)

        trn_acc = [history.history['categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
        val_acc = [history.history['val_categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
        trn_los = history.history['loss']
        val_los = history.history['val_loss']

        print(trn_acc[-1])
        print(trn_los[-1])
        print(val_acc[-1])
        print(val_los[-1])
        print(results)

        plt.clf()
        plt.plot(trn_acc)
        plt.plot(val_acc)
        plt.title(f'(SGD) Accuracy with regularization= l%d and a = %.3f' % (RCN[reg2], RC[reg2]))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
        plt.savefig('(SGD) Accuracy with regularization= l%d and a = %.3f.png' % (RCN[reg2], RC[reg2]))

        plt.clf()
        plt.plot(trn_los)
        plt.plot(val_los)
        ax = plt.gca()
        ax.set_ylim([0, 10])
        plt.title(f'(SGD) Losses with regularization= l%d and a = %.3f' % (RCN[reg2], RC[reg2]))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['Training Losses', 'Validation Losses'], loc='upper right')
        plt.savefig('(SGD) Losses with regularization= l%d and a = %.3f.png' % (RCN[reg2], RC[reg2]))

    # 5 SGD optimizer with L1
    tf.random.set_seed(89)
    print('For regularization choice = l1-norm and a = 0.01')
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu', kernel_regularizer=REGULAR_L1[0]))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_regularizer=REGULAR_L1[0]))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax', kernel_regularizer=REGULAR_L1[0]))
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])
    history = model.fit(train_x, train_y, batch_size=256, epochs=EPOCHS, validation_data=(val_x, val_y), verbose=0)
    results = model.evaluate(test_x, test_y)

    trn_acc = [history.history['categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
    val_acc = [history.history['val_categorical_accuracy'][i] * 100 for i in range(EPOCHS)]
    trn_los = history.history['loss']
    val_los = history.history['val_loss']

    print(trn_acc[-1])
    print(trn_los[-1])
    print(val_acc[-1])
    print(val_los[-1])
    print(results)

    plt.clf()
    plt.plot(trn_acc)
    plt.plot(val_acc)
    plt.title('(SGD) Accuracy with regularization choice is l1-norm with a = 0.01')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right')
    plt.savefig('(SGD) Accuracy with regularization choice is l1-norm with a = 0.01.png')

    plt.clf()
    plt.plot(trn_los)
    plt.plot(val_los)
    plt.title('(SGD) Losses with regularization choice is l1-norm with a = 0.01')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper right')
    plt.savefig('(SGD) Losses with regularization choice is l1-norm with a = 0.01.png')
