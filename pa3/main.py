'''
Author: Caleb Ehrisman
Course- Advanced AI CSC-549
Assignment - Programming Assignment #3 - Mountain Car

Tasks
 - Implement Sarsa(lambda) to solve mountain car problem
 - Use Linear Function Approximation with Fourier Basis functions
 - Show different learning curves for 3rd, 5th, and 7th order Fourier bases
 - Create surface plot of the value function
 - Answer short response question
'''

import gym
from agent import Agent
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
    parse gathers command line arguments.

    
    :return: a list of all parsed arguments
    """
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Specify trained model to start with')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    if args.checkpoint is not None:
        env = gym.make("MountainCar-v0", render_mode="human")

    else:
        env = gym.make("MountainCar-v0")

    env.action_space.seed(1000)

    agent = Agent(env)
    rewards, max_pos = agent.learn(env, 1000)

    num_completed = sum([1 if m > 0.5 else 0 for m in max_pos])
    print(f'{num_completed} success out of {1000} attempts')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Reward values")
    ax.set(xlim=(0, 1000), xticks=np.arange(0, 1000, 200))
    pd.Series(rewards).plot(kind='line')
    plt.show()



