import numpy as np
import matplotlib.pyplot as plt

def plot_hist(probs, idx):
    # probs = np.load('./probs_train_weighted2.npz')['arr_0']
    # probs = np.load('./reprojection_errors.npz')['arr_0']
    # probs = 1 - (probs - np.min(probs))/np.ptp(probs)
    # rewards = np.load('./full_dataset.npz')['arr_0']
    rewards = np.load('./full_dataset_test.npz')['arr_0']
    neg_rewards = 1 - rewards

    rew1 = np.multiply(probs, rewards)
    rew0 = np.multiply(probs, neg_rewards)

    rew1_non_zero = list(rew1[np.nonzero(rew1)])
    rew0_non_zero = list(rew0[np.nonzero(rew0)])

    plt.hist(rew1_non_zero, bins=10, alpha=0.5, label='Reward=1')
    plt.hist(rew0_non_zero, bins=10, alpha=0.5, label="Reward=0")
    plt.legend()
    plt.ylabel("Count")
    plt.xlabel("CNN Out")
    plt.savefig(f'cnn_out_test.jpg')
    plt.cla()

def plot_hist_from_file():
    probs = np.load('./probs_train_weighted.npz')['arr_0']
    # probs = np.load('./reprojection_errors.npz')['arr_0']
    # probs = 1 - (probs - np.min(probs))/np.ptp(probs)
    rewards = np.load('./full_dataset.npz')['arr_0']
    neg_rewards = 1 - rewards

    rew1 = np.multiply(probs, rewards)
    rew0 = np.multiply(probs, neg_rewards)

    rew1_non_zero = list(rew1[np.nonzero(rew1)])
    rew0_non_zero = list(rew0[np.nonzero(rew0)])

    plt.hist(rew0_non_zero, bins=10, alpha=1, label="Reward=0")
    plt.hist(rew1_non_zero, bins=10, alpha=1, label='Reward=1')
    plt.legend()
    plt.ylabel("Count")
    plt.xlabel("Re-projection Error")
    # plt.show()
    plt.savefig(f'cnn_hist.jpg', dpi=800)

    plt.cla()

if __name__ == '__main__':
    plot_hist_from_file()

