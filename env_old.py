import numpy as np
import sys


class Env(object):
    """ The transportation network. """

    R = 5  # number of regions
    N = 1000  # number of cars
    H = 360  # horizon in minute
    num_slots = 3
    len_slot = 120  # in minute

    # Passenger arrival rate (number per minute)
    # This decides the car initial distribution
    lambda_by_region = np.asarray([[0.108, 0.108, 0.108, 0.108, 1.08], \
                                   [0.72, 0.48, 0.48, 0.48, 0.12], \
                                   [0.12, 0.12, 0.12, 1.32, 0.12]])  # per car, per hour
    lambda_by_region = lambda_by_region * N / 60.  # all cars, per minute
    lambda_max = lambda_by_region.max(axis=0)  # max over time
    accept_prob = lambda_by_region / lambda_max[np.newaxis, :]
    # for converting time-inhomogeneous Poisson process into a time-homogeneous one
    dest_prob = np.asarray([
        [[.6, .1, 0, .3, 0], \
         [.1, .6, 0, .3, 0], \
         [0, 0, .7, .3, 0], \
         [.2, .2, .2, .2, .2], \
         [.3, .3, .3, .1, 0]], \
        [[.1, 0, 0, .9, 0], \
         [0, .1, 0, .9, 0], \
         [0, 0, .1, .9, 0], \
         [.05, .05, .05, .8, .05], \
         [0, 0, 0, .9, .1]], \
        [[.9, .05, 0, .05, 0], \
         [.05, .9, 0, .05, 0], \
         [0, 0, .9, .1, 0], \
         [.3, .3, .3, .05, .05], \
         [0, 0, 0, .1, .9]]
    ])
    L = 5  # candidate patience time in minute

    # Mean travel time in minute
    tau = np.asarray([[[0.15, 0.25, 1.25, 0.2, 0.4], \
                       [0.25, 0.1, 1.1, 0.1, 0.3], \
                       [1.25, 1.1, 0.1, 1, 0.65], \
                       [0.25, 0.15, 1, 0.15, 0.25], \
                       [0.5, 0.4, 0.75, 0.25, 0.2]], \
                      [[0.15, 0.25, 1.25, 0.2, 0.4], \
                       [0.25, 0.1, 1.1, 0.1, 0.3], \
                       [1.25, 1.1, 0.1, 1, 0.65], \
                       [0.2, 0.1, 1, 0.15, 0.25], \
                       [0.4, 0.3, 0.65, 0.25, 0.2]], \
                      [[0.15, 0.25, 1.25, 0.2, 0.4], \
                       [0.25, 0.1, 1.1, 0.1, 0.3], \
                       [1.25, 1.1, 0.1, 1, 0.65], \
                       [0.2, 0.1, 1, 0.15, 0.25], \
                       [0.4, 0.3, 0.65, 0.25, 0.2]]])
    tau = (np.ceil(np.array(tau) * 60)).astype('int')
    tau_max = tau.max(axis=1).max(axis=0)  # max travel time to a region
    car_dims = 1 + tau_max + L  # free cars
    car_dims_cum = np.insert(car_dims.cumsum(), 0, 0)
    stay_empty_nby_car_dim = (1 + L) * R
    # Nearby cars associated with each region, assigned to "stay empty" upon reaching destination
    # so its region will not change, neither will its total distance to the region;
    # before reaching the destination, this car is not free to be assigned a trip anymore!
    car_dim = car_dims_cum[-1] + stay_empty_nby_car_dim
    # car_dim = car_dims.sum()
    psg_dim = R * R
    obs_dim = car_dim + psg_dim  # Observation dimension excluding time component

    # Rewards
    # Car-passenger matching
    c = np.ones((num_slots, R, R), dtype=float)
    # Empty-car routing
    tilde_c = np.zeros((num_slots, R, R), dtype=float)

    def __init__(self):

        # Ride requests
        self.queue = None  # all day
        self.next_ride = None  # index
        # self.all_rewards = None  # total number of ride requests over the horizon
        self.car_init_dist = None

    @staticmethod
    def min_to_slot(minute):
        """ Minute to slot
         minute in [0, H)
         Passenger arrivals, etc., by minute """

        return min(int(minute / Env.len_slot), Env.num_slots - 1)
        # min((minute - 1) // Env.len_slot, Env.num_slots - 1)

    def reset(self, dest_known=True):
        """ Reset everything at the start of an episode """

        # Ride requests
        queue_by_region = [[] for r in range(Env.R)]
        for r in range(Env.R):
            tpast = 0
            while True:
                interarriv = np.random.exponential(scale=1. / Env.lambda_max[r], size=None)
                tpast += interarriv
                if tpast >= Env.H:
                    break
                queue_by_region[r].append(tpast)
        queue_inhomo = [[] for r in range(Env.R)]
        for r in range(Env.R):
            for event in queue_by_region[r]:
                idx = Env.min_to_slot(event)
                if np.random.choice(a=2, size=None, p=[1 - Env.accept_prob[idx, r], Env.accept_prob[idx, r]]) > 0.5:
                    queue_inhomo[r].append(event)
        del queue_by_region
        self.queue = []
        if dest_known:
            for r in range(Env.R):
                for event in queue_inhomo[r]:
                    idx = Env.min_to_slot(event)
                    self.queue.append([event, r, np.random.choice(a=Env.R, size=None, p=Env.dest_prob[idx, r])])
        else:
            for r in range(Env.R):
                for event in queue_inhomo[r]:
                    self.queue.append([event, r])
        self.queue.sort(key=lambda x: x[0])  # sort by arrival time
        self.next_ride = 0

        # Car initial distribution
        init_car_dist = Env.lambda_by_region[0].copy()
        init_car_dist = init_car_dist / init_car_dist.sum() * Env.N
        surplus = init_car_dist - init_car_dist.astype('int')
        surplus_from_largest = surplus.argsort()[::-1]
        total_surplus = int(surplus.sum())

        init_car_dist = init_car_dist.astype('int')
        i, j = 0, 0  # sliding window
        while i < len(surplus) and total_surplus > 0:
            while j < len(surplus) and abs(surplus[surplus_from_largest[j]] - surplus[surplus_from_largest[i]]) < 1e-8:
                j += 1
            probs = np.asarray([self.lambda_by_region[0][surplus_from_largest[k]] for k in range(i, j)])
            probs = probs / probs.sum()
            remaining_cars = np.random.multinomial(min(j - i, total_surplus), probs)
            for k in range(i, j):
                init_car_dist[surplus_from_largest[k]] += remaining_cars[k - i]
            total_surplus -= min(j - i, total_surplus)
        self.car_init_dist = init_car_dist

    def get_initial_state(self):
        """ Get initial state """

        if self.car_init_dist is None:
            self.reset()  # Sometimes self.car_init_dist is not None but still resetting maybe needed!
        s_cp = np.zeros(self.obs_dim, dtype=int)  # unscaled
        for reg in range(self.R):
            s_cp[self.car_dims_cum[reg]] = self.car_init_dist[reg]
        # passengers
        s_t = 0
        while self.next_ride < len(self.queue) and (self.queue[self.next_ride][0] < s_t + 1):
            s_cp[self.car_dim + self.queue[self.next_ride][1] * self.R + self.queue[self.next_ride][2]] += 1
            self.next_ride += 1
        return s_cp, s_t

    def num_avb_nby_cars(self, state_vector):
        """ Number of available (a.k.a. free) nearby cars """

        num = 0
        for reg in range(self.R):
            num += state_vector[self.car_dims_cum[reg]:(self.car_dims_cum[reg] + self.L + 1)].sum()
        return num

    def get_next_state(self, s_post, dec_epoch):
        """ Get the next state from the post-decision state of the last candidate
        s_post updated in-place
        """
        s_post[self.car_dim:] = 0  # Passengers unattended leave the transportation network.
        next_dec_epoch = dec_epoch + 1
        # print('Next decision epoch: ', next_dec_epoch)
        if next_dec_epoch < self.H:
            # Passengers accumulation.
            while self.next_ride < len(self.queue) and self.queue[self.next_ride][0] < next_dec_epoch + 1:
                # print('Next ride index: ', self.next_ride)
                # print('Time of arrival: ', self.queue[self.next_ride][0])
                s_post[self.car_dim + self.queue[self.next_ride][1] * self.R + self.queue[self.next_ride][2]] += 1
                self.next_ride += 1
            # Car dynamics.
            # 1) Free cars.
            for reg in range(self.R):
                s_post[self.car_dims_cum[reg]] += s_post[self.car_dims_cum[reg] + 1]
            for reg in range(self.R):
                s_post[(self.car_dims_cum[reg] + 1):(self.car_dims_cum[reg + 1] - 1)] = \
                    s_post[(self.car_dims_cum[reg] + 2):self.car_dims_cum[reg + 1]]
            for reg in range(self.R):
                s_post[self.car_dims_cum[reg + 1] - 1] = 0
            # 2) Nearby cars assigned to stay empty at destination.
            for reg in range(self.R):  # Assigned car -> free car!
                s_post[self.car_dims_cum[reg]] += s_post[self.car_dims_cum[-1] + reg * (1 + self.L)] + \
                                                  s_post[self.car_dims_cum[-1] + reg * (1 + self.L) + 1]
                s_post[self.car_dims_cum[-1] + reg * (1 + self.L)] = 0
                # s_post[self.car_dims_cum[-1] + reg * (1 + self.L)] += \
                #   s_post[self.car_dims_cum[-1] + reg * (1 + self.L) + 1]
            for reg in range(self.R):  # assigned car still on the way
                s_post[(self.car_dims_cum[-1] + reg * (1 + self.L) + 1):
                       (self.car_dims_cum[-1] + reg * (1 + self.L) + self.L)] = \
                    s_post[(self.car_dims_cum[-1] + reg * (1 + self.L) + 2):
                           (self.car_dims_cum[-1] + reg * (1 + self.L) + 1 + self.L)]
            for reg in range(self.R):
                s_post[self.car_dims_cum[-1] + reg * (1 + self.L) + self.L] = 0
        else:  # next_dec_epoch == self.H
            s_post[:] = np.nan  # terminal state

        return next_dec_epoch


def test():
    print('To do (tentative): \n'
          '1. State; 2. Transition model; 3. Processing a candidate given action probabilities from policy network;')


if __name__ == '__main__':
    test()
