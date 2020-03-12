import tensorflow as tf
from tensorflow import keras.backend as K

# Actor model defined using Keras

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, lrate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.lrate = lrate

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = tf.keras.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = tf.keras.layers.Dense(units=32, activation='elu')(states)
        #net = layers.BatchNormalization()(net)
        net = tf.keras.layers.Dense(units=64, activation='elu')(net)
        #net = layers.BatchNormalization()(net)
        net = tf.keras.layers.Dense(units=32, activation='elu')(net)
        #net = layers.BatchNormalization()(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = tf.keras.layers.Dense(units=self.action_size, activation='tanh',
            name='raw_actions')(net)

        # Scale [-1, 1] output for each action dimension to proper range
        actions = tf.keras.layers.multiply(raw_actions, action_range/2)

        # Create Keras model
        self.model = tf.keras.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = tf.keras.layers.Input(shape=(self.action_size,))
        loss = tf.keras.backend.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = tf.keras.optimizers.Adam(lr=self.lrate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = tf.keras.backend.function(
            inputs=[self.model.input, action_gradients, tf.keras.backend.learning_phase()],
            outputs=[],
            updates=updates_op)

# Critic model defined in Keras
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, lrate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.lrate = lrate

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = tf.keras.layers.Input(shape=(self.state_size,), name='states')
        actions = tf.keras.layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = tf.keras.layers.Dense(units=32, activation='elu')(states)
        #net_states = layers.BatchNormalization()(net_states)
        net_states = tf.keras.layers.Dense(units=64, activation='elu')(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = tf.keras.layers.Dense(units=32, activation='elu')(actions)
        #net_actions = layers.BatchNormalization()(net_actions)
        net_actions = tf.keras.layers.Dense(units=64, activation='elu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = tf.keras.layers.Add()([net_states, net_actions])
        net = tf.keras.layers.Activation('elu')(net)
        #net = layers.BatchNormalization()(net)

        # Add more layers to the combined network if needed
        net = tf.keras.layers.Dense(units=8, activation='elu')(net)
        #net = layers.BatchNormalization()(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = tf.keras.layers.Dense(units=1, name='q_values', kernel_regularizer=tf.keras.regularizers.l2(0.01))(net)

        # Create Keras model
        self.model = tf.keras.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = tf.keras.optimizers.Adam(lr=self.lrate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = tf.keras.backend.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = tf.keras.backend.function(
            inputs=[*self.model.input, tf.keras.backend.learning_phase()],
            outputs=action_gradients)
