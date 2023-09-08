import time
import tensorflow as tf
from keras.callbacks import Callback

def build_model(wts_path, train=False, to_save_as=False, model_path=None):
    if model_path:
        return tf.keras.models.load_model(model_path)

    my_model = get_model()

    if wts_path:
        my_model.load_weights(wts_path)

    if train:
        class myCallback(Callback):
            def on_epoch_end(self, epoch, logs={}):
                if logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.95:
                    print('Stopping training')
                    my_model.stop_training = True

        callbacks = myCallback()
        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        my_model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

        if wts_path:
            my_model.save_weights('{}-{}'.format(wts_path, round(time.time())))
        else:
            my_model.save_weights(to_save_as)

    return my_model

def get_model():
    model = tf.keras.Sequential([])
    model.add(tf.keras.layers.InputLayer(input_shape=(32, 32, 1)))
    model.add(tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (2, 2), activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Conv2D(256, (2, 2), activation="relu", padding="same"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(9, activation="softmax"))
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model