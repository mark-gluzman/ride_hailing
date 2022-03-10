# Pre-training
# First round: Get scale, offset for state through 100*10 episodes

# Second round: Get scale, offset for value network output..........

import numpy as np
import pandas as pd
from time import time
# from env_old import Env
# from policy_old import Policy
from env import Env
from policy import Policy
from value_function import NNValueFunction
from scaler import Scaler
import sys


def main():
    assert len(sys.argv) >= 2 and int(sys.argv[1]) == 1 or int(sys.argv[1]) == 2, \
        'No or incorrect command line argument for round number!'

    round_num = int(sys.argv[1])

    if round_num == 1:
        # First round
        start_time = time()
        env = Env()
        policy_func = Policy(obs_dim=env.obs_dim, act_dim=env.R * env.R,
                             hid1_mult=1, hid3_mult=5, sz_voc=env.H, embed_dim=env.num_slots * 2, reg_str=5e-3, temp=2.)
        num_episodes = 100 * 10  # 100 actors, each running 10 episodes
        expanding_mean, expanding_var, num_samples = \
            np.zeros(env.obs_dim, dtype=float), np.zeros(env.obs_dim, dtype=float), 0  # expanding_var <- Welford
        matching_rates = np.zeros(num_episodes)  # side product
        for i_episode in range(num_episodes):
            trajectory = policy_func.run_episode(env, scaler=None)
            state = np.asarray(trajectory['state'])

            new_mean = state.mean(axis=0)
            new_expanding_mean = (expanding_mean * num_samples + new_mean * len(state)) / (num_samples + len(state))
            if num_samples <= 0:  # first pass
                expanding_var = state.var(axis=0)
            else:
                new_mean_sq = np.square(new_mean)
                new_var = state.var(axis=0)
                expanding_var = (num_samples * (expanding_var + np.square(expanding_mean)) + len(state) * (
                        new_var + new_mean_sq)) / (num_samples + len(state)) - np.square(new_expanding_mean)
            expanding_var = np.maximum(0., expanding_var)  # occasionally goes negative, clip
            expanding_mean = new_expanding_mean
            num_samples += len(state)

            matching_rates[i_episode] = trajectory['matching_rate']
        suffix = '_action_reinterpreted'
        pd.DataFrame((np.asarray([range(env.obs_dim), expanding_mean, np.sqrt(expanding_var)])).T,
                     columns=['dimension', 'mean', 'std']).to_csv('../statistics/scaler/state' + suffix + '.csv')
        pd.DataFrame((np.asarray([range(num_episodes), matching_rates])).T, columns=['episode', 'matching_rate']). \
            to_csv('../statistics/matching_rate/policy_network_no_training_no_standardization' + suffix + '.csv')
        print('Average car-passenger matching rate over {} episodes: {:.2f}%...'.
              format(num_episodes, matching_rates.mean()))
        print()
        run_time = (time() - start_time) / 60.
        print('Elapsed time: {:.2f} minute...'.format(run_time))
        print()

    # Second round
    if round_num == 2:
        print('Second round...')
        start_time = time()
        env = Env()
        policy_func = Policy(obs_dim=env.obs_dim, act_dim=env.R * env.R,
                             hid1_mult=1, hid3_mult=5, sz_voc=env.H, embed_dim=env.num_slots * 2, reg_str=5e-3, temp=2.)
        num_episodes = 100 * 10  # 100 actors, each running 10 episodes
        scaler = Scaler('state_action_reinterpreted')
        val_func = NNValueFunction(obs_dim=env.obs_dim, hid1_mult=1, hid3_mult=5, sz_voc=env.H,
                                   embed_dim=env.num_slots * 2)
        expanding_mean, expanding_var, num_samples = np.zeros(1, dtype=float), np.zeros(1, dtype=float), 0.
        matching_rates = np.zeros(num_episodes)  # side product
        for i_episode in range(num_episodes):
            trajectory = policy_func.run_episode(env, scaler=scaler)
            x = np.asarray(trajectory['state_scaled'])
            x_t = np.asarray(trajectory['state_time'])

            y = val_func.predict(x, x_t)
            # print('y: ', type(y))  # np.ndarray
            # print('y shape: ', y.shape)  # (5...., )
            # print('x shape: ', x.shape)  # (5...., 393)
            # print(x[1050][80:90])
            # print()
            # print(x_t[1050])
            # print()
            # print(y[2050])
            y = y[:, np.newaxis]

            new_mean = y.mean(axis=0)
            new_expanding_mean = (expanding_mean * num_samples + new_mean * len(y)) / (num_samples + len(y))
            if num_samples <= 0:  # first pass
                expanding_var = y.var(axis=0)
            else:
                new_mean_sq = np.square(new_mean)
                new_var = y.var(axis=0)
                expanding_var = (num_samples * (expanding_var + np.square(expanding_mean)) + len(y) *
                                 (new_var + new_mean_sq)) / (num_samples + len(y)) - np.square(new_expanding_mean)
            expanding_var = np.maximum(0., expanding_var)  # occasionally goes negative, clip
            expanding_mean = new_expanding_mean
            num_samples += len(y)

            matching_rates[i_episode] = trajectory['matching_rate']
            print('Finished Episode {}...'.format(i_episode))

        suffix = '_action_reinterpreted'
        pd.DataFrame((np.asarray([range(1), expanding_mean, np.sqrt(expanding_var)])).T,
                     columns=['dimension', 'mean', 'std']). \
            to_csv('../statistics/scaler/value_network_output_no_training' + suffix + '.csv')
        # print('Expanding mean: {:.2f}...'.format(expanding_mean))
        # print('Expanding std: {:.2f}...'.format(np.sqrt(expanding_var)))
        # print()
        pd.DataFrame((np.asarray([range(num_episodes), matching_rates])).T, columns=['episode', 'matching_rate']). \
            to_csv('../statistics/matching_rate/policy_network_no_training' + suffix + '.csv')

        print('Average car-passenger matching rate over {} episodes: {:.2f}%...'.
              format(num_episodes, matching_rates.mean()))
        print()
        run_time = (time() - start_time) / 60.
        print('Elapsed time: {:.2f} minute...'.format(run_time))
        print()


if __name__ == '__main__':
    main()
