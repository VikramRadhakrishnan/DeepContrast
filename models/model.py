import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Input, Conv2D, Concatenate, BatchNormalization, Activation, Flatten
from tensorflow.keras import Model
from tensorflow.keras import regularizers

tf.keras.backend.set_floatx('float64')

# Actor model defined using Keras

class Actor:
    """Deep Q Model."""

    def __init__(self, state_size, action_size, name="Actor"):
        """Initialize parameters.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            name (string): Name of the model
        """
        # Initialize the state and action dimensions
        self.state_size = state_size
        self.action_size = action_size
        self.name = name
        
        # Build the actor model
        self.build_model()

    def build_model(self):
        ''' Build the Neural Net for Deep-Q learning Model.
        Convolutional layers with Batch Norm followed by dense layers.'''
        # Define input layer (states)
        states = Input(shape=self.state_size)

        # Input layer is 25x25x2
        net = Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(states)
        net = BatchNormalization(fused=False)(net)

        # Now 25x25x8
        actions = Conv2D(1, (3, 3), strides=(1, 1), activation='tanh', padding='same', kernel_initializer='glorot_uniform')(net)

        # Create Keras model
        self.model = Model(inputs=states, outputs=actions, name=self.name)

# Critic model defined in Keras
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, name="Critic"):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Dimensions of 1st hidden layer
            fc2_units (int): Dimensions of 2nd hidden layer
            name (string): Name of the model
        """

        # Initialize the state and action dimensions
        self.state_size = state_size
        self.action_size = action_size
        self.name = name

        # Build the critic model
        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = Input(shape=self.state_size)
        actions = Input(shape=self.action_size)

        # Add hidden layer for state pathway
        net_states = Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(states)
        # Now 25x25x8

        # Combine state and action pathways
        net = Concatenate(axis=-1)([net_states, actions])
        # Now 12x12x9

        # Add more layers to the combined network
        net = Conv2D(8, (3, 3), strides=(1, 1), activation='relu', padding='same', kernel_initializer='glorot_uniform')(net)
        net = BatchNormalization(fused=False)(net)
        net = Conv2D(8, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='glorot_uniform')(net)
        net = BatchNormalization(fused=False)(net)

        # Flatten to 1D
        net = Flatten()(net) 
        
        # Add final output layer to produce action values (Q values)
        Q_values = Dense(units=1, activation='linear', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0001))(net)

        # Create Keras model
        self.model = Model(inputs=[states, actions], outputs=Q_values, name=self.name)


