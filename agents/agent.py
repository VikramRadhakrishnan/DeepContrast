from models.model import Actor, Critic
from utils.noise_model import OUNoise, GaussianNoise
from utils.replay_buffer import ReplayBuffer

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

import numpy as np

# Deep Deterministic Policy Gradients Agent
class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, state_size, action_size, actor_lr, critic_lr,
                 random_seed, mu, theta, sigma, buffer_size, batch_size,
                 epsilon_start, epsilon_min, epsilon_decay,
                 gamma, tau, n_time_steps, n_learn_updates, device):

        self.state_size = state_size
        self.action_size = action_size
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, name="Actor_local")
        self.actor_target = Actor(state_size, action_size, name="Actor_target")
        self.actor_optimizer = Adam(learning_rate=self.actor_lr)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, name="Critic_local")
        self.critic_target = Critic(state_size, action_size, name="Critic_target")
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)
        
        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = GaussianNoise(action_size, random_seed, mu, sigma)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Replay memory
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, random_seed)

        # Algorithm parameters
        self.gamma = gamma                     # discount factor
        self.tau = tau                         # for soft update of target parameters
        self.n_time_steps = n_time_steps       # number of time steps before updating network parameters
        self.n_learn_updates = n_learn_updates # number of updates per learning step

        # Device
        self.device = device

        tf.keras.backend.clear_session()

    def reset(self):
        """Reset the agent."""
        self.noise.reset()

    def step(self, time_step, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state[:], action[:], reward, next_state[:], done)

        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
        if time_step % self.n_time_steps != 0:
            return

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            
            # Train the network for a number of epochs specified by the parameter
            for i in range(self.n_learn_updates):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = np.expand_dims(state, axis=0)
        action = self._act_tf(state)
        action = action.numpy()[0]

        if add_noise:
            action += np.random.normal(0.0, 0.1, action.shape) * self.epsilon

        # Clip action between +1 and -1
        action = action.clip(-1, 1)

        return action

    @tf.function
    def _act_tf(self, state):
        return self.actor_local.model(state)

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences : tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        return self._learn_tf(experiences, tf.constant(self.gamma, dtype=tf.float64))

    @tf.function
    def _learn_tf(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        rewards = tf.expand_dims(rewards, 1)
        dones = tf.expand_dims(dones, 1)

        # ---------------------------- update critic ---------------------------- #
        with tf.GradientTape() as tape:
            tape.watch(self.critic_local.model.trainable_variables)
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target.model(next_states)
            Q_targets_next = self.critic_target.model([next_states, actions_next])
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            Q_targets = tf.stop_gradient(Q_targets)
            # Compute critic loss
            Q_expected = self.critic_local.model([states, actions])
            critic_loss = MSE(Q_expected, Q_targets)
        
        # Minimize the loss
        critic_grad = tape.gradient(critic_loss, self.critic_local.model.trainable_variables)

        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_local.model.trainable_variables))

        # ---------------------------- update actor ---------------------------- #
        with tf.GradientTape() as tape:
            tape.watch(self.actor_local.model.trainable_variables)
            # Compute actor loss
            actions_pred = self.actor_local.model(states)
            actor_loss = -tf.reduce_mean(self.critic_local.model([states, actions_pred]))

        # Minimize the loss
        actor_grad = tape.gradient(actor_loss, self.actor_local.model.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_local.model.trainable_variables))

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local.model, self.critic_target.model, self.tau)
        self.soft_update(self.actor_local.model, self.actor_target.model, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: TF2 model
            target_model: TF2 model
            tau (float): interpolation parameter 
        """
        for target_var, local_var in zip(target_model.weights, local_model.weights):
            target_var.assign(tau * local_var + (1.0 - tau) * target_var)

class TD3(DDPG):
    def __init__(self, state_size, action_size, actor_lr, critic_lr,
                 random_seed, mu, theta, sigma, buffer_size, batch_size,
                 gamma, tau, n_time_steps, n_learn_updates, device,
                 actor_update_freq):
        super(TD3, self).__init__(state_size, action_size, actor_lr, critic_lr,
                                  random_seed, mu, theta, sigma, buffer_size, batch_size,
                                  gamma, tau, n_time_steps, n_learn_updates, device)

        # Critic Network #2 (w/ Target Network)
        self.critic2_local = Critic(state_size, action_size, name="Critic2_local")
        self.critic2_target = Critic(state_size, action_size, name="Critic2_target")

        self.train_step = 0
        self.actor_update_freq = actor_update_freq

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        target_next_actions = actor_target(next_state) + e
        Q'_min - min(critic_target1, critic_target2)(next_state, target_next_actions)
        Q_targets = r + γ * Q'_min
        where:
            actor_target(state) -> action
            critic_target1(state, action) -> Q-value1
            critic_target2(state, action) -> Q-value2
        Params
        ======
            experiences : tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            e: noise
        """
        self.train_step += 1
        self._learn_tf(experiences, tf.constant(self.gamma, dtype=tf.float64))

    @tf.function
    def _learn_tf(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        with tf.GradientTape(persistent=True) as tape:
            # Get predicted next-state actions and Q values from target models
            actions_next = self.actor_target.model(next_states)
            actions_next += tf.clip_by_value(tf.random.normal(shape=tf.shape(actions_next), mean=0.0, stddev=0.01, dtype=tf.float64), -0.01, 0.01)
            Q1 = self.critic_target.model([next_states, actions_next])
            Q2 = self.critic2_target.model([next_states, actions_next])
            Q_targets_next = tf.math.minimum(Q1, Q2)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            # Compute critic loss
            Q1_expected = self.critic_local.model([states, actions])
            Q2_expected = self.critic2_local.model([states, actions])
            critic_loss = MSE(Q1_expected, Q_targets) + MSE(Q2_expected, Q_targets)
        
        # Minimize the loss
        critic1_grad = tape.gradient(critic_loss, self.critic_local.model.trainable_variables)
        critic2_grad = tape.gradient(critic_loss, self.critic2_local.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic1_grad, self.critic_local.model.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic2_grad, self.critic2_local.model.trainable_variables))

        if self.train_step % self.actor_update_freq:
            # ---------------------------- update actor ---------------------------- #
            with tf.GradientTape() as tape:
                # Compute actor loss
                actions_pred = self.actor_local.model(states)
                actor_loss = -tf.reduce_mean(self.critic_local.model([states, actions_pred]))

            # Minimize the loss
            actor_grad = tape.gradient(actor_loss, self.actor_local.model.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_local.model.trainable_variables))

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local.model, self.critic_target.model, self.tau)
            self.soft_update(self.critic2_local.model, self.critic_target.model, self.tau)
            self.soft_update(self.actor_local.model, self.actor_target.model, self.tau)


