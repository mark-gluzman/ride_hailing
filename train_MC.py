import ray  # package for distributed computations
import numpy as np
from policy import Policy
from value_function import NNValueFunction
from utils import Logger, Scaler
import os
import argparse
import processingNetwork as pn
import random
import datetime
import copy
ray.init(temp_dir='/tmp/ray2')

MAX_ACTORS = 50  # max number of parallel simulations

def diag_dot(A, B):
    # returns np.diag(np.dot(A, B))
    return np.einsum("ij,ji->i", A, B)


def run_policy(network_id, policy, scaler, logger, gamma,
               policy_iter_num, skipping_steps, cycles_num, episodes, time_steps):
    """
    Run given policy and collect data
    :param network_id: queuing network structure and first-order info
    :param policy: queuing network policy
    :param scaler: normalization values
    :param logger: metadata accumulator
    :param gamma: discount factor
    :param policy_iter_num: policy iteration
    :param skipping_steps: number of steps when action does not change  ("frame-skipping" technique)
    :param episodes: number of parallel simulations (episodes)
    :param time_steps: max time steps in an episode
    :return: trajectories = (states, actions, rewards)
    """

    total_steps = 0
    action_optimal_sum = 0
    total_zero_steps = 0

    burn = 1

    scale, offset = scaler.get()

    '''
    initial_states_set = random.sample(scaler.initial_states, k=episodes)
    trajectories, total_steps, action_optimal_sum, total_zero_steps, array_actions = policy.run_episode(ray.get(network_id), scaler, time_steps, cycles_num, skipping_steps,  initial_states_set[0])
    '''

    #### declare actors for distributed simulations of a current policy#####
    remote_network = ray.remote(Policy)
    simulators = [remote_network.remote(obs_dim=ray.get(network_id).obs_dim, act_dim=ray.get(network_id).R * ray.get(network_id).R,
                         hid1_mult=1, hid3_mult=5, sz_voc=ray.get(network_id).H, embed_dim=ray.get(network_id).num_slots * 2, reg_str=5e-3, temp=2.) for _ in range(MAX_ACTORS)]
    actors_per_run = episodes // MAX_ACTORS # do not run more parallel processes than number of cores
    remainder = episodes - actors_per_run * MAX_ACTORS
    weights = policy.get_weights()  # get neural network parameters
    ray.get([s.set_weights.remote(weights) for s in simulators]) # assign the neural network weights to all actors
    ######################################################



    ######### save neural network parameters to file ###########
    file_weights = os.path.join(logger.path_weights, 'weights_'+str(policy_iter_num)+'.npy')
    np.save(file_weights, weights)
    ##################

    scaler_id = ray.put(scaler)
    initial_states_set = random.sample(scaler.initial_states, k=episodes)  # sample initial states for episodes

    ######### policy simulation ########################
    accum_res = []  # results accumulator from all actors
    trajectories = []  # list of trajectories
    for j in range(actors_per_run):
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network_id, scaler_id, time_steps, cycles_num,
                                skipping_steps,  initial_states_set[j*MAX_ACTORS+i]) for i in range(MAX_ACTORS)]))
    if remainder>0:
        accum_res.extend(ray.get([simulators[i].run_episode.remote(network_id, scaler_id, time_steps, cycles_num,
                                skipping_steps, initial_states_set[actors_per_run*MAX_ACTORS+i]) for i in range(remainder)]))
    print('simulation is done')

    for i in range(len(accum_res)):
        trajectories.append(accum_res[i][0])
        total_steps += accum_res[i][1]  # total time-steps
        action_optimal_sum += accum_res[i][2]  # absolute number of actions consistent with the "optimal policy"
        total_zero_steps += accum_res[i][3]  # absolute number of states for which all actions are optimal
    #################################################


    optimal_ratio = action_optimal_sum / (total_steps * skipping_steps)  # fraction of actions that are optimal
    # fraction of actions that are optimal excluding transitions when all actions are optimal
    pure_optimal_ratio = (action_optimal_sum - total_zero_steps)/ (total_steps * skipping_steps - total_zero_steps)

    average_reward = np.mean(np.concatenate([t['rewards'] for t in trajectories]))


    #### normalization of the states in data ####################
    unscaled = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    if gamma < 1.0:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]


    else:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]
            z = t['rewards'] - average_reward
            t['rewards'] = z
    ##################################################################





    scaler.update_initial(np.hstack((unscaled, np.zeros(len(unscaled))[np.newaxis].T)))

    ########## results report ##########################
    print('Average cost: ',  -average_reward)

    logger.log({'_AverageReward': -average_reward,
                'Steps': total_steps,
                'Zero steps':total_zero_steps,
                '% of optimal actions': int(optimal_ratio * 1000) / 10.,
                '% of pure optimal actions': int(pure_optimal_ratio * 1000) / 10.,
    })
    ####################################################
    return trajectories

def add_disc_sum_rew(trajectories, policy, network, gamma, lam, scaler, iteration):
    """
    compute value function for further training of Value Neural Network
    :param trajectory: simulated data
    :param network: transportation network
    :param policy: current policy
    :param gamma: discount factor
    :param lam: lambda parameter in GAE
    :param scaler: normalization values
    """
    start_time = datetime.datetime.now()
    for trajectory in trajectories:


        if gamma < 1:
            #advantages = discount(x=tds_pi,   gamma=lam*gamma, v_last = tds_pi[-1]) - tds_pi + tds_a   # advantage function
            disc_sum_rew = discount(x=trajectory['rewards'],   gamma= gamma, v_last = trajectory['rewards'][-1])
        else:
            #advantages = relarive_af(unscaled_obs, td_pi=tds_pi, td_act=tds_a, lam=lam)  # advantage function
            disc_sum_rew = relarive_af(trajectory['unscaled_obs'], td_pi=trajectory['rewards'], lam=1)  # advantage function


        #trajectory['advantages'] = np.asarray(advantages)
        trajectory['disc_sum_rew'] = disc_sum_rew


    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_disc_sum_rew time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

    burn = 1

    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])
    if iteration ==1:
        scaler.update(np.hstack((unscaled_obs, disc_sum_rew)))
    scale, offset = scaler.get()
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    disc_sum_rew_norm = (disc_sum_rew - offset[-1]) * scale[-1]
    if iteration ==1:
        for t in trajectories:
            t['observes'] = (t['unscaled_obs'] - offset[:-1]) * scale[:-1]

    return observes, disc_sum_rew_norm

def discount(x, gamma, v_last):
    """ Calculate discounted forward sum of a sequence at each point """
    disc_array = np.zeros((len(x), 1))
    disc_array[-1] = v_last
    for i in range(len(x) - 2, -1, -1):
        if x[i+1]!=0:
            disc_array[i] = x[i] + gamma * disc_array[i + 1]

    return disc_array


def relarive_af(unscaled_obs, td_pi,  lam):
    # return advantage function
    disc_array = np.copy(td_pi)
    sum_tds = 0
    for i in range(len(td_pi) - 2, -1, -1):
        if np.sum(unscaled_obs[i+1]) != 0:
            sum_tds = td_pi[i+1] + lam * sum_tds
        else:
            sum_tds = 0
        disc_array[i] += sum_tds

    return disc_array



def add_value(trajectories, val_func, scaler):
    """
    # compute value function from the Value Neural Network
    :param trajectory_whole: simulated data
    :param val_func: Value Neural Network
    :param scaler: normalization values
    """
    start_time = datetime.datetime.now()
    scale, offset = scaler.get()


    # approximate value function for trajectory_whole['unscaled_obs']
    for trajectory in trajectories:
        values = val_func.predict(trajectory['observes'])
        trajectory['values'] = values / scale[-1] + offset[-1]


    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('add_value time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')

def build_train_set(trajectories, gamma, scaler):
    """
    # data pre-processing for training
    :param trajectory_whole:  simulated data
    :param scaler: normalization values
    :return: data for further Policy and Value neural networks training
    """



    for trajectory in trajectories:
        values = trajectory['values']

        unscaled_obs = trajectory['unscaled_obs']
        advantages = trajectory['rewards'] - values + gamma * np.append(values[1:], values[-1])
        trajectory['advantages'] = np.asarray(advantages)




    start_time = datetime.datetime.now()
    burn = 1


    unscaled_obs = np.concatenate([t['unscaled_obs'][:-burn] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'][:-burn] for t in trajectories])

    scale, offset = scaler.get()
    actions = np.concatenate([t['actions'][:-burn] for t in trajectories])
    advantages = np.concatenate([t['advantages'][:-burn] for t in trajectories])
    observes = (unscaled_obs - offset[:-1]) * scale[:-1]
    advantages = advantages  / (advantages.std() + 1e-6) # normalize advantages



    # ########## averaging value function estimations over all data ##########################
    # states_sum = {}
    # states_number = {}
    # states_positions = {}
    #
    # for i in range(len(unscaled_obs)):
    #     if tuple(unscaled_obs[i]) not in states_sum:
    #         states_sum[tuple(unscaled_obs[i])] = disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] = 1
    #         states_positions[tuple(unscaled_obs[i])] = [i]
    #
    #     else:
    #         states_sum[tuple(unscaled_obs[i])] +=  disc_sum_rew[i]
    #         states_number[tuple(unscaled_obs[i])] += 1
    #         states_positions[tuple(unscaled_obs[i])].append(i)
    #
    # for key in states_sum:
    #     av = states_sum[key] / states_number[key]
    #     for i in states_positions[key]:
    #         disc_sum_rew[i] = av
    # ########################################################################################
    end_time = datetime.datetime.now()
    time_policy = end_time - start_time
    print('build_train_set time:', int((time_policy.total_seconds() / 60) * 100) / 100., 'minutes')
    return observes,  actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    # metadata tracking

    time_total = datetime.datetime.now() - logger.time_start
    logger.log({'_mean_act': np.mean(actions),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode,
                '_time_from_beginning_in_minutes': int((time_total.total_seconds() / 60) * 100) / 100.
                })

# TODO: check shadow name
def main(network_id, num_policy_iterations, gamma, lam, kl_targ, batch_size, hid1_mult, episode_duration,
         clipping_parameter, skipping_steps):
    """
    # Main training loop
    :param: see ArgumentParser below
    """

    now = datetime.datetime.utcnow().strftime("%b-%d_%H-%M-%S")  # create unique directories
    time_start= datetime.datetime.now()
    logger = Logger(logname=ray.get(network_id).network_name, now=now, time_start=time_start)


    val_func = NNValueFunction(obs_dim=400, hid1_mult=1, hid3_mult=5, sz_voc=360, embed_dim=6) # Value Neural Network initialization
    policy = Policy(obs_dim=ray.get(network_id).obs_dim, act_dim=ray.get(network_id).R * ray.get(network_id).R,
                         hid1_mult=1, hid3_mult=5, sz_voc=ray.get(network_id).H, embed_dim=ray.get(network_id).num_slots * 2, reg_str=5e-3, temp=2.) # Policy Neural Network initialization

    iteration = 0  # count of policy iterations
    weights_set = []
    scaler_set = []
    while iteration < num_policy_iterations:
        # decrease clipping_range and learning rate
        iteration += 1
        alpha = 1. - iteration / num_policy_iterations
        policy.clipping_range = max(0.01, alpha*clipping_parameter)
        policy.lr_multiplier = max(0.05, alpha)

        print('Clipping range is ', policy.clipping_range)

        trajectories = run_policy(network_id, policy, scaler, logger, gamma, iteration, skipping_steps, cycles_num,
                                      episodes=batch_size, time_steps=episode_duration) #simulation

        add_value(trajectories, val_func, scaler)  # add estimated values to episodes
        observes, disc_sum_rew_norm = add_disc_sum_rew(trajectories, policy, ray.get(network_id), gamma, lam, scaler, iteration)  # calculate values from data

        val_func.fit(observes, np.arrange(1, ray.get(network_id).H), disc_sum_rew_norm, logger)  # update value function
        add_value(trajectories, val_func, scaler)  # add estimated values to episodes
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories, gamma, scaler)


        #scale, offset = scaler.get()
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, iteration)  # add various stats
        policy.update(observes, actions, np.squeeze(advantages), logger)  # update policy


        #print('V(0):', disc_sum_rew[0], val_func.predict([observes[0]])[0][0]/ scale[-1] + offset[-1])

        logger.write(display=True)  # write logger results to file and stdout

    weights = policy.get_weights()

    file_weights = os.path.join(logger.path_weights, 'weights_' + str(iteration) + '.npy')
    np.save(file_weights, weights)

    file_scaler = os.path.join(logger.path_weights, 'scaler_' + str(iteration) + '.npy')
    scale, offset = scaler.get()
    np.save(file_scaler, np.asarray([scale, offset]))
    weights_set.append(policy.get_weights())
    scaler_set.append(copy.copy(scaler))


    logger.close()
    policy.close_sess()
    val_func.close_sess()




if __name__ == "__main__":
    network = Env()

    network_id = ray.put(network)


    parser = argparse.ArgumentParser(description=('Train policy for a transportation network '
                                                  'using Proximal Policy Optimizer'))

    parser.add_argument('-n', '--num_policy_iterations', type=int, help='Number of policy iterations to run',
                        default = 200)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor',
                        default = 1)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default = 1)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default = 0.003)
    parser.add_argument('-b', '--batch_size', type=int, help='Number of episodes per training batch',
                        default = 50)
    parser.add_argument('-m', '--hid1_mult', type=int, help='Size of first hidden layer for value and policy NNs',
                        default = 10)
    parser.add_argument('-t', '--episode_duration', type=int, help='Number of time-steps per an episode',
                        default = 360)
    parser.add_argument('-c', '--clipping_parameter', type=float, help='Initial clipping parameter',
                        default = 0.2)
    parser.add_argument('-s', '--skipping_steps', type=int, help='Number of steps for which control is fixed',
                        default = 1)



    args = parser.parse_args()
    main(network_id,  **vars(args))