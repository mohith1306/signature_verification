from tensorflow.keras import layers, Model, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class AbsoluteDifference(layers.Layer):
    def call(self, inputs):
        x, y = inputs
        return tf.math.abs(x - y)
    def get_config(self):
        return super().get_config()

def build_siamese_model(input_shape=(128, 128, 1)):
    def build_base_network(input_shape):
        model = Sequential([
            # First block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu')
        ])
        return model

    base_network = build_base_network(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    feat_a = base_network(input_a)
    feat_b = base_network(input_b)

    distance = AbsoluteDifference(name="abs_diff")([feat_a, feat_b])
    # Add more layers after distance
    x = layers.Dense(64, activation='relu')(distance)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-5),  # Lower learning rate
        metrics=['accuracy']
    )
    return model
