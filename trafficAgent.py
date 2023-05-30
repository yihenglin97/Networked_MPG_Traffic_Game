import numpy as np
import trafficEnv
import math
from scipy import special


class TrafficAgent:
    def __init__(self, state_num, action_num, time_max, random_init=False):
        self.state_num = state_num
        self.action_num = action_num
        self.time_max = time_max

        # We store two policy tables for the time-invariant policy and the time-varying policy
        # Both use softmax parameterization
        self.invariant_policy = np.zeros((state_num, action_num))
        if random_init:
            self.invariant_policy = np.random.normal(size=(self.state_num, self.action_num))
        self.tv_policy = np.zeros((time_max, state_num, action_num))

    def reset(self):
        self.invariant_policy = np.zeros((state_num, action_num))
        self.tv_policy = np.zeros((time_max, state_num, action_num))

    # sample an action under the current policy
    # expect pair = (time, state), if time-invariant, set time = -1
    def sample_action(self, pair):
        time, state = pair
        # get the softmax policy parameters
        params = self.invariant_policy[state, :]
        if time != -1:
            params = self.tv_policy[time, state, :]
        # compute the probability vector
        prob_vec = special.softmax(params)
        # randomly select an action based on prob_vec
        action = np.random.choice(a=self.action_num, p=prob_vec) - 1
        return action

    # compute the stationary distribution under the current policy
    # requires a TrafficNetwork object, initial distribution, and is_invariant as argument
    def stationary_dist(self, network, init_dist, horizon, is_invariant):
        result = np.zeros((horizon, self.state_num))
        result[0, :] = init_dist
        for t in range(1, horizon):
            for s in range(self.state_num):
                params = self.invariant_policy[s, :]
                if not is_invariant:
                    params = self.tv_policy[t, s, :]
                prob_vec = special.softmax(params)
                for a in range(self.action_num):
                    next_state = network.get_neighbor(s, a-1)
                    result[t, next_state] += (prob_vec[a] * result[t-1, s])
        return result



if __name__ == '__main__':
    state_num = 2
    action_num = 2
    time_max = 10
    agent = TrafficAgent(state_num, action_num, time_max)
    action_list = []
    for t in range(time_max):
        action_list.append(agent.sample_action((t, 0)))
    print("action history: {}".format(action_list))

    network = trafficEnv.TrafficNetwork(node_num=2)
    network.add_edge((0, 1))
    stationary_dist = agent.stationary_dist(network, np.array([1, 0]), 8, True)
    print("stationary distribution: \n{}".format(stationary_dist))

