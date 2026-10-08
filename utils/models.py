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