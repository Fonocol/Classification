import tensorflow as tf
from tensorflow.keras import layers, models

def build_mlp(input_shape=(784,), hidden_units=[128, 64], num_classes=10, learning_rate=0.001):

    model = models.Sequential()
    
    # Couche d'entrée
    model.add(layers.Input(shape=input_shape))
    
    # Couches cachées
    for units in hidden_units:
        model.add(layers.Dense(units, activation='relu'))

    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compilation : Catégorielle cross-entropy + descente de gradient (Adam ou SGD)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_mlp_cifar(learning_rate=0.0005):
    model = models.Sequential([
        # Couche d'entrée
        layers.Input(shape=(3072,)),
        
        # Bloc 1 (512 neurones)
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Bloc 2 (256 neurones)
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.25),
        
        # Bloc 3 (128 neurones)
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Couche de sortie
        layers.Dense(10, activation='softmax')
    ])
    
    # Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model




def build_lenet5_mnist(learning_rate=0.001):
    """
    Architecture LeNet-5 adaptée pour MNIST 28x28x1.
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),

        # Sortie : 10 classes
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model




def build_lenet5_cifar10(learning_rate=0.001, use_dropout=True):
    """
    Architecture LeNet-5 adaptée pour CIFAR-10 (entrée : 32x32x3).
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))


    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation='relu'))  # devient 10x10
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # devient 5x5

    # Aplatissement pour le passage aux couches denses
    model.add(layers.Flatten())

   
    model.add(layers.Dense(120, activation='relu'))
    if use_dropout:
        model.add(layers.Dropout(0.2))

  
    model.add(layers.Dense(84, activation='relu'))
    if use_dropout:
        model.add(layers.Dropout(0.2))

 
    model.add(layers.Dense(10, activation='softmax'))


    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model