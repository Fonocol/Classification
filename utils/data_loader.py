import tensorflow as tf

def load_mnist(flatten=False, one_hot=True):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    if flatten:
        X_train = X_train.reshape((-1, 28 * 28))
        X_test = X_test.reshape((-1, 28 * 28))
    else:
        #CNN 
        X_train = X_train[..., None]
        X_test = X_test[..., None]

    if one_hot:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (X_train, y_train), (X_test, y_test)


def load_cifar10(flatten=False, one_hot=True):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    if flatten:
        X_train = X_train.reshape((-1, 32 * 32 * 3))
        X_test = X_test.reshape((-1, 32 * 32 * 3))

    if one_hot:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    return (X_train, y_train), (X_test, y_test)