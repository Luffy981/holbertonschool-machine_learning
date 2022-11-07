#!/usr/bin/env python3
"""
Training an agent that can play Atari's Breakout
"""


import gym
import numpy as np
from tensorflow.keras.layers import Input, Permute, Conv2D, \
        Flatten, Dense
from tensorflow.keras import Model, optimizers
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor

# Frames
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    """
    Defines Atari environment to play Breakout
    """
    def process_observation(self, observation):
        """
        Resizes images and makes grayscale to conserve memory
        """


        # observation > Tuple (nd.array, dict)
        # print("SHAPE", observation[0].ndim)
        # assert observation[0].shape[1] == 3 # (height, width, channel)
        # get image from array
        image = Image.fromarray(observation[0])
        # resize image and convert to grayscale
        # sampling filter > Image.ANTIALIAS = Resampling.LANCZOS = 1
        image = image.resize((84, 84), Image.LANCZOS).convert('L')
        # put back into array
        processed_observation = np.array(image)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """
        Converts a batch of images to float32
        """
        processed_batch = batch.astype('float32') / 255.0
        return processed_batch
    
    def process_reward(self, reward):
        """
        Processes reward between -1 and 1
        """
        return np.clip(reward, -1., 1.)


def create_CNN_model(number_actions, input_shape=(84, 84)):
    """
    Creates a CNN model with Keras as defined by the DeepMind resource
    """
    # Each state consists of 4 frames, each with input_shape=(84x84)
    # input_shape comes from copped pixels to save memory
    full_input_shape = (WINDOW_LENGTH, ) + input_shape
    inputs = Input(shape=full_input_shape)
    layer_0 = Permute((2, 3, 1))(inputs)

    # First hidden layer convolves 32 8x8 filters with stride 4
    layer_1 = Conv2D(filters=32,
                     kernel_size=8,
                     strides=4,
                     activation='relu',
                     data_format='channels_last')(layer_0)
    # Second hidden layer convolver 64 4x4 filters with stride 2
    layer_2 = Conv2D(filters=64,
                     kernel_size=4,
                     strides=2,
                     activation='relu',
                     data_format='channels_last')(layer_1)
    # third hidden layer convolves 64 3x3 filters with stride 1
    layer_3 = Conv2D(filters=64,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     data_format='channels_last')(layer_2)
    # Fourth hidden layer flattens the third layer
    layer_4 = Flatten()(layer_3)
    # fifth hidden layer applies relu activation with 512 units
    layer_5 = Dense(units=512,
                           activation='relu')(layer_4)
    # Output layer is fully-conected linear layer with
    # simgle output for each valid action
    outputs = Dense(units=number_actions,
                           activation='linear')(layer_5)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def training():
    """
    Trains an agent to play Atari's Breakout
    """
    env = gym.make('ALE/Breakout-v5')
    # env.reset()
    nb_actions = env.action_space.n
    model = create_CNN_model(nb_actions)
    model.summary()
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                                  attr='eps',
                                  value_max=1.0,
                                  value_min=0.1,
                                  value_test=0.05,
                                  nb_steps=1000000)
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   policy=policy,
                   memory=memory,
                   processor=processor,
                   nb_steps_warmup=50000,
                   gamma=0.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.0)
    dqn.compile(optimizers.Adam(lr=0.00025),
                metrics=['mae'])
    dqn.fit(env,
            nb_steps=4000000,
            log_interval=25000,
            visualize=False,
            verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
    env.close()


if __name__=='__main__':
        training()
