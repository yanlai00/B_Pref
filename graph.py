import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Normalized rewards
folder_prefix = './exp/quadruped_walk/H1024_L2_lr0.0001/teacher_b1_g1_m0_s0_e0/label_smooth_0.0/schedule_0/PEBBLE_init1000_unsup9000_inter30000_maxfeed2000_seg50_acttanh_Rlr0.0003_Rbatch200_Rupdate50_en3_sampleNone_large_batch10_seed'
def rewards_from_seed_list(folder_prefix, seeds):
    # Assume the same number of steps for each run
    steps = None
    ep_rewards = []
    for seed in seeds:
        train_file = folder_prefix+str(seed)+'/train.csv'
        train_data = pd.read_csv(train_file)
        ep_rewards.append(train_data['episode_reward'])
        steps = train_data['step']
    return steps, np.mean(ep_rewards, axis=0), np.std(ep_rewards, axis=0)

l2_seeds = [92834, 38483, 48397, 20910, 32948]
steps, rew, rew_std = rewards_from_seed_list(folder_prefix, l2_seeds)
plt.plot(steps/1000000, rew)
plt.fill_between(steps/1000000, rew-rew_std, rew+rew_std, alpha=.5)
plt.xlabel('Environment Step x10^6')
plt.ylabel('Episode Return')
plt.title('L2 Regularization of Reward Function under Noisy Teacher')
plt.savefig('l2_graph.png')

