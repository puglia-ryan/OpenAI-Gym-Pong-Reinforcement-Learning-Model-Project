from tensorflow import keras

class CustomModel:
    def __init__(self, input_shape, actions):
        self.input_shape = input_shape
        self.actions = actions
        self.model = self.build_model()

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=self.input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.actions, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

