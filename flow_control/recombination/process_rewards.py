import numpy as np
import matplotlib.pyplot as plt

# rewards_prod = np.mean(np.load('./recombination/rewards_final_prod/rewards_prod_all.npz')['arr_0'], axis=0)
# rewards_sum = np.mean(np.load('./recombination/rewards_final/rewards_sum_all.npz')['arr_0'], axis=0)
rewards_min = np.mean(np.load('./recombination/rewards_abs/rewards_min_all.npz')['arr_0'], axis=0)
rewards_min1 = np.mean(np.load('./recombination/rewards_final/rewards_min_all.npz')['arr_0'], axis=0)
# rewards_ml = np.mean(np.load('./recombination/rewards_ml/rewards_ml1.npz')['arr_0'], axis=0)

x = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
# plt.plot(x, rewards_prod, ".-", label='error_fn=prod')
# plt.plot(x, rewards_sum, ".-", label='error_fn=sum')
plt.plot(x, rewards_min, ".-", label='error_fn=updated')
plt.plot(x, rewards_min1, ".-", label='error_fn=old')
# plt.plot(x, rewards_ml, ".-", label='error_fn=ML')

plt.plot([0, 75], [0.205, 0.205], "k--")

plt.xlabel("#Recordings")
plt.ylabel("Mean Rewards")
plt.title("Recombination (Debug Mode)")
plt.legend()
plt.savefig('rewards_abs.jpg')
