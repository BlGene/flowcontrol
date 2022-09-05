import numpy as np
import json
import matplotlib.pyplot as plt


def plot_rewards(reward_file, savefig, title):
    fp = open(reward_file)
    rewards = json.load(fp)

    total_rew = None

    for seed, val in rewards.items():
        if total_rew is None:
            total_rew = val
        else:
            total_rew = [x + y for x, y in zip(total_rew, val)]

    keys = list(rewards.keys())

    length = len(rewards[keys[0]])
    x = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

    total_rew = [rew / len(keys) for rew in total_rew]

    mean = 0.152
    std = 0.09

    # mean = 0.205
    # std = 0.102

    plt.plot([0, 70], [mean, mean], "k--")
    plt.axhspan(mean - std, mean + std, facecolor='0.5', alpha=0.5)

    plt.plot(x, total_rew, ".-")
    plt.xlabel("#Recordings")
    plt.ylabel("Mean Rewards")
    plt.title(title)
    plt.savefig(savefig, dpi=800)


def main():
    reward_file = './old_files/old_reward_folders/rewards_trapeze_steps_2/reward_ss.json'
    savefig = "./old_files/old_reward_folders/rewards_trapeze_steps_2/reward_ss.png"
    title = 'Shape Sorting task Baseline'

    plot_rewards(reward_file, savefig, title)

main()
