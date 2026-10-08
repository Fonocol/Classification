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