# from env_old import Env
# from policy_old import Policy
from value_function import NNValueFunction

from env import Env
from policy import Policy
from scaler import Scaler


def main():
    env = Env()
    policy_func = Policy(obs_dim=env.obs_dim, act_dim=env.R * env.R, hid1_mult=1, hid3_mult=5, sz_voc=env.H,
                         embed_dim=env.num_slots * 2)  # Tune: kl_targ
    trajectory = policy_func.run_episode(env, scaler=None)
    # print('Actions: \n', trajectory['action'][-50:])
    val_func = NNValueFunction(obs_dim=env.obs_dim, hid1_mult=1, hid3_mult=5, sz_voc=env.H, embed_dim=env.num_slots * 2)

    #  Pre-training:
    #  1) Get offset, scale for state through 100*10 episodes (hid1_mult=1, hid3_mult=5, reg_str=5e-3, temp=2)

    #  2) Get offset, scale for value network output...

    scaler = Scaler('state_action_reinterpreted')
    trajectory = policy_func.run_episode(env, scaler=scaler)


if __name__ == '__main__':
    main()

