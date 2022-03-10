import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Embedding, Flatten
from tensorflow.python.keras.regularizers import l2
import ray.experimental
from collections import OrderedDict
import sys


class Policy(object):
    """ Policy neural network """

    def __init__(self, obs_dim, act_dim, hid1_mult, hid3_mult, sz_voc, embed_dim, reg_str=5e-3, kl_targ=np.inf,
                 clipping_range=0.2, temp=2.0):
        """
        :param obs_dim: num observation dimensions
        :param act_dim: num action dimensions
        :param kl_targ: target KL divergence between pi_old and pi_new
        :param hid1_mult: size of first hidden layer, multiplier of obs_dim + embed_dim
        :param clipping_range:
        :param temp: temperature parameter
        """
        self.obs_dim = obs_dim  # not including time component
        self.act_dim = act_dim
        self.hid1_mult = hid1_mult
        self.hid3_mult = hid3_mult
        self.sz_voc = sz_voc
        self.embed_dim = embed_dim
        self.reg_str = reg_str

        self.beta = 3  # dynamically adjusted D_KL loss multiplier
        self.kl_targ = kl_targ  # A large KL target implies no early stopping
        self.clipping_range = clipping_range
        self.temp = temp

        self.epochs = 3
        self.lr_multiplier = 1.0  # dynamically adjust lr when D_KL out of control
        self.lr = None

        self._build_graph()

    def _build_graph(self):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn()
            self._logprob()
            self._kl_entropy()
            self._loss_train_op()
            self._loss_initial_op()
            self.init = tf.compat.v1.global_variables_initializer()
            self.sess = tf.compat.v1.Session(graph=self.g)
            self.variables = ray.experimental.TensorFlowVariables(self.loss, self.sess)
            self.sess.run(self.init)

    def _placeholders(self):
        """ Define placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.compat.v1.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.time_ph = tf.compat.v1.placeholder(tf.int32, (None,), 'time')
        self.act_ph = tf.compat.v1.placeholder(tf.int32, (None,), 'act')  # 1 queueing station

        self.advantages_ph = tf.compat.v1.placeholder(tf.float32, (None,), 'advantages')
        # strength of D_KL loss terms:
        self.beta_ph = tf.compat.v1.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.compat.v1.placeholder(tf.float32, (), 'eta')
        self.lr_ph = tf.compat.v1.placeholder(tf.float32, (), 'eta')  # learning rate:

        self.old_act_prob_ph = tf.compat.v1.placeholder(tf.float32, (None, self.act_dim), 'old_act_prob')

    def _policy_nn(self):
        """ Neural Network architecture for policy approximation function
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        hid1_size = (self.obs_dim + self.embed_dim) * self.hid1_mult  # 10 empirically determined
        hid3_size = self.hid3_mult  # 10 empirically determined
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # heuristic to set learning rate based on NN size (tuned on 'Hopper-v1')
        self.lr = 5e-4  # 9e-4 empirically determined
        # 3 hidden layers with tanh activations
        embed = Embedding(self.sz_voc, self.embed_dim, name='simple_embedding',
                          embeddings_regularizer=l2(self.reg_str), trainable=True)(self.time_ph)
        embed = Flatten()(embed)
        out = tf.layers.dense(tf.concat([self.obs_ph, embed], axis=1), hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / (self.obs_dim + self.embed_dim))), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3")
        act_prob_out = tf.layers.dense(tf.divide(out, self.temp), self.act_dim, tf.nn.softmax,
                                       kernel_initializer=tf.random_normal_initializer(
                                           stddev=np.sqrt(1 / hid3_size)), name="act_prob")
        self.act_prob_out = act_prob_out

    def _logprob(self):
        """
        Calculate probabilities using previous step's model parameters and new parameters being trained.
        """
        # probabilities of actions which agent took with policy
        self.act_probs = tf.reduce_sum(
            self.act_prob_out * tf.one_hot(indices=self.act_ph, depth=self.act_prob_out.shape[1]), axis=1)
        # probabilities of actions which agent took with old policy
        self.act_probs_old = tf.reduce_sum(
            self.old_act_prob_ph * tf.one_hot(indices=self.act_ph, depth=self.old_act_prob_ph.shape[1]), axis=1)

    def _kl_entropy(self):
        """
        Calculate KL-divergence between old and new distributions
        """

        self.entropy = 0
        self.kl = 0

        kl = tf.reduce_sum(self.act_prob_out * (tf.math.log(tf.clip_by_value(self.act_prob_out, 1e-10, 1.0))
                                                - tf.math.log(tf.clip_by_value(self.old_act_prob_ph, 1e-10, 1.0))),
                           axis=1)
        entropy = tf.reduce_sum(self.act_prob_out * tf.math.log(tf.clip_by_value(self.act_prob_out, 1e-10, 1.0)),
                                axis=1)

        self.entropy += -tf.reduce_mean(entropy, axis=0)  # sum of entropy of pi(obs)
        self.kl += tf.reduce_mean(kl, axis=0)

    def _loss_train_op(self):
        """
        Calculate the PPO loss function
        """

        ratios = tf.exp(tf.math.log(self.act_probs) - tf.math.log(self.act_probs_old))
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clipping_range,
                                          clip_value_max=1 + self.clipping_range)
        loss_clip = tf.minimum(tf.multiply(self.advantages_ph, ratios), tf.multiply(self.advantages_ph, clipped_ratios))
        # Our objective is to maximize loss_clip
        self.loss = -tf.reduce_mean(loss_clip)  # - self.entropy*0.0001
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def _loss_initial_op(self):
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr_ph)
        self.train_init = optimizer.minimize(self.kl)

    def sample(self, obs, x_t, stochastic=True):
        """
        :param obs: state
        :param x_t: time component
        :param stochastic: stochastic or deterministic
        :return: if stochastic=True returns pi(a|x), else returns distribution with prob=1 on argmax[ pi(a|x) ]
        """

        feed_dict = {self.obs_ph: obs, self.time_ph: x_t}
        if stochastic:
            return self.sess.run(self.act_prob_out, feed_dict=feed_dict)
        else:
            pr = self.sess.run(self.act_prob_out, feed_dict=feed_dict)
            inx = np.argmax(pr)
            ar = np.zeros(self.act_dim, dtype=float)
            ar[inx] = 1
            return ar[np.newaxis]

    def update(self, observes, times, actions, advantages, logger):
        """
        Policy Neural Network update
        :param observes: states
        :param times: time components
        :param actions: actions
        :param advantages: estimation of antantage function at observed states
        :param logger: statistics accumulator
        """
        feed_dict = {self.obs_ph: observes,
                     self.time_ph: times,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.beta_ph: self.beta,
                     self.lr_ph: self.lr * self.lr_multiplier}
        old_act_prob_np = self.sess.run(self.act_prob_out, feed_dict)  # actions probabilities w.r.t the current policy
        feed_dict[self.old_act_prob_ph] = old_act_prob_np

        loss = 0
        kl = 0
        entropy = 0
        for e in range(self.epochs):  # training
            self.sess.run(self.train_op, feed_dict)
            loss, kl, entropy = self.sess.run([self.loss, self.kl, self.entropy], feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                print('Early stopping: D_KL diverges badly!')
                break

        # actions probabilities w.r.t the new and old (current) policies
        act_probs, act_probs_old = self.sess.run([self.act_probs, self.act_probs_old], feed_dict)
        ratios = np.exp(np.log(act_probs) - np.log(act_probs_old))
        if self.clipping_range is not None:
            clipping_range = self.clipping_range
        else:
            clipping_range = 0

        logger.log({'PolicyLoss': loss,
                    'Clipping': clipping_range,
                    'Max ratio': max(ratios),
                    'Min ratio': min(ratios),
                    'Mean ratio': np.mean(ratios),
                    'PolicyEntropy': entropy,
                    'KL': kl,
                    'Beta': self.beta,
                    '_lr_multiplier': self.lr_multiplier})

    def run_episode(self, network, scaler):
        """
        One episode simulation
        :param network: transportation network
        :param scaler: standardized values
        :return: collected data
        """
        # print('For now, forget about the standardization!!!!!!!!!')
        # print()
        scale, offset = 1., 0.
        if scaler is not None:
            pass
        observes_unscaled, observes, times, actions, rewards = [], [], [], [], []

        network.reset()  # Needed; because the input network may have run for a while,
        # and thus network.car_init_dist is not None but resetting is needed!
        s_running, dec_epoch = network.get_initial_state()  # unscaled
        # print('Initial car, passenger state: ', s_running)
        # print()

        while dec_epoch < network.H:
            # print('Decision epoch: ', dec_epoch)
            # slot_id = network.min_to_slot(dec_epoch)

            # Find a sequence of candidates:
            # num_psg = s_running[network.car_dim:].sum()
            num_free_nby_cars = network.num_avb_nby_cars(s_running)
            # number of nearby cars NOT assigned to "stay empty at destination"
            # print('Number of passengers: ', num_psg)
            # print('Number of nearby cars: ', num_free_nby_cars)
            # break
            # while num_psg + num_free_nby_cars > 0.5:

            while num_free_nby_cars > 0.5:  # cars' perspective
                s_scaled = (s_running - offset)/scale
                act_prob_on_trip = self.sample([s_scaled], [dec_epoch], stochastic=True)[0]
                # print('Action prob type: ', type(act_prob_on_trip))  # np.ndarray
                # print('Action prob shape: ', act_prob_on_trip.shape) # (1, 25)
                # print('Action prob: \n', act_prob_on_trip)
                # sys.exit('...testing.')
                while True:  # resample until feasible action
                    trip_type = np.random.choice(self.act_dim, p=act_prob_on_trip)
                    orig, dest = divmod(trip_type, network.R)
                    eta_w_car = np.where(s_running[network.car_dims_cum[orig]:
                                                   (network.car_dims_cum[orig] + network.L + 1)] > 0.5)[0]
                    if len(eta_w_car) <= 0:
                        continue  # no free nearby car associated with this orig region -> resample
                    # eta_closest = eta_w_car[0] # closest car's distance
                    slot_id = network.min_to_slot(dec_epoch + eta_w_car[0])
                    # Collect state:
                    observes_unscaled.append(s_running.copy())  # because s_running will change
                    observes.append(s_scaled.copy())
                    # Collect state - time component:
                    times.append(dec_epoch)
                    # Collect action
                    actions.append(trip_type)  # matching or routing can be inferred from the reward
                    if s_running[network.car_dim + trip_type] > 0.5:
                        # print('Car-passenger matching!')
                        rewards.append(network.c[slot_id][orig, dest])
                        # post-decision state
                        s_running[network.car_dim+trip_type] -= 1
                        s_running[network.car_dims_cum[dest] + eta_w_car[0] + network.tau[slot_id][orig, dest]] += 1
                    else:
                        # print('Empty-car routing!')
                        rewards.append(network.tilde_c[slot_id][orig, dest])
                        # post-decision state
                        if dest != orig:
                            s_running[network.car_dims_cum[dest] + eta_w_car[0] + network.tau[slot_id][orig, dest]] += 1
                        else:  # Action "staying empty at the destination":
                            # same total distance to the ASSIGNED NEXT destination, but not available anymore
                            s_running[network.car_dims_cum[-1] + dest * (1 + network.L) + eta_w_car[0]] += 1
                    # post-decision state
                    s_running[network.car_dims_cum[orig]+eta_w_car[0]] -= 1  # One fewer available nearby car
                    num_free_nby_cars -= 1  # processed one free nearby car
                    # s_running, dec_epoch is the post-decision state now
                    break
                # print('Rewards: ', rewards[-1])
                # sys.exit('...testing!')
            # next state for the last candidate:
            # update s_running
            # s_running =
            # dec_epoch += 1  # will become s_{t,1} for the next decision epoch
            # print('Post-decision state for the last candidate: ')
            # print('Number of free nearby cars: ', network.num_avb_nby_cars(s_running))
            # print('Number of nearby cars assigned to stay empty at destination: ',
            # s_running[network.car_dims_cum[-1]:network.car_dim].sum())
            # print('Number of cars: ', s_running[:network.car_dim].sum())
            # print('Number of passengers: ', s_running[network.car_dim:].sum())
            dec_epoch = network.get_next_state(s_running, dec_epoch)  # Passengers unattended leave!
            assert dec_epoch >= network.H or s_running[:network.car_dim].sum() == network.N, 'Car number problematic!'
            # print('Next decision epoch: ', dec_epoch)
            # print('Number available nearby cars: ', network.num_avb_nby_cars(s_running))
            # sys.exit('...testing!!!')
            # observes[-1] =
            # observes_unscaled[-1] =
            # times[-1] =
        # terminal state, s_{H+1, 1} - do not need to derive it.
        trajectory = OrderedDict([('state', observes_unscaled), ('state_scaled', observes), ('state_time', times),
                                  ('action', actions), ('reward', rewards), ('number_passengers', len(network.queue)),
                                  ('matching_rate', float(sum(rewards)) / len(network.queue) * 100)])
        print()
        print('Car-passenger matching rate: {:.2f}%...'.format(trajectory['matching_rate']))
        print()
        return trajectory


def test():
    policy_func = Policy(obs_dim=100, act_dim=25, hid1_mult=1, hid3_mult=5, sz_voc=360, embed_dim=6, reg_str=5e-3,
                         kl_targ=1)

    print('Changes: \n'
          '1) act_dim is NOT a list of action dimensions for multiple queueing stations; \n'
          '2) In the constructor, the order of hid1_mult and kl_targ is swapped; \n'
          '3) hid1_mult (from 10 to 1); \n'
          '4) hid3_mult (from 10 to 5); \n'
          '5) Added the time component; \n'
          '6) Incorporated time component embedding; \n')


if __name__ == '__main__':
    test()

