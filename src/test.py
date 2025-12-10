from envs.doors.doors import Doors
import numpy as np

if __name__ == "__main__":

    env = Doors(n_agents = 2, mishear_prob = 0.1, comm_error = 0.1,
                 rew_c = 10, pen_l = 0.5, episode_limit = 100, state_is_obs = False, state_obs_concat = False)

    obs, state = env.reset()
    print(state, obs)

    for i in range(20):
        (env.step(np.array([0,0,0,0])))
        print(env.get_obs())
        print(env.get_state())

    print(env.step(np.array([0,1,0,0])))