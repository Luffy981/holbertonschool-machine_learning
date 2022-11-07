#!/usr/bin/env python3
"""
display a game played by the agent trained by train.py
"""

import gym
from tensorflow.keras import optimizers
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

AtariProcessor = __import__('train').AtariProcessor
create_CNN_model = __import__('train').create_CNN_model
# Frames
WINDOW_LENGTH = 4

def playing():
    """
    display a game played by the agent trained by train.py
    """
    env = gym.make('ALE/Breakout-v5')
    env.reset()
    nb_actions = env.action_space.n
    model = create_CNN_model(nb_actions)
    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    processor = AtariProcessor()
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   processor=processor,
                   memory=memory)
    dqn.compile(optimizers.Adam(lr=0.00025),
                metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env,
             nb_episodes=10,
             visualize=True)
if __name__ == "__main__":
    playing()

