#This file is the blueprint for the neural network which the agent will use
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Permute, Conv2D
#from keras.initializers import he_normal


class CustomModel:
    def __init__(self, input_shape, actions):
        self.input_shape = input_shape
        self.actions = actions
        self.model = self.build_model()

    def build_model(self):
        """
        model = keras.Sequential([
            keras.layers.Input(shape=self.input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(128, input_dim=12*84*84, activation='relu'),
            keras.layers.Dense(self.actions, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        """

        model = Sequential()
        #These layers will help the model recognise what is going on on the screen.
        #We lower the stride length for each layer
        model.add(Permute((2, 3, 1), input_shape=self.input_shape))
        model.add(Conv2D(32, (8, 8), strides=(4, 4), kernel_initializer="he_normal"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), kernel_initializer="he_normal"))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (2, 2), strides=(1, 1), kernel_initializer="he_normal"))
        model.add(Activation('relu'))

        #The layers above give us a multidimentional array, so we need to flatten these for our intended usage
        model.add(Flatten())

        model.add(Dense(500))
        model.add(Dense(1000))
        model.add(Activation('relu'))
        model.add(Dense(self.actions))
        model.add(Activation('linear'))

        return model

    def load_model_func(self):
        self.model = keras.models.load_model("DQN_CHECKPOINT.hf5")