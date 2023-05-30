import numpy as np
import trafficEnv
import math
from scipy import special
import trafficAgent
from tqdm import trange
import matplotlib.pyplot as plt

# give a global state in nd_array
# compute the global state code

def global_state_encoder(global_state, state_num):
    agent_num = global_state.shape[0]
    global_state_code = 0
    for i in range(agent_num):
        global_state_code *= state_num
        global_state_code += global_state[i]
    return global_state_code

# give a global_state_code, which is a nonnegative integer
# compute the global state in nd_array

def global_state_decoder(global_state_code, state_num, agent_num):
    global_state = np.zeros(agent_num, dtype=int)
    code = global_state_code
    for i in range(agent_num):
        global_state[agent_num-1-i] = code % state_num
        code = code//state_num
    return global_state

class CentralizedOptimizer:
    def __init__(self, network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon, epsilon=0.5):
        self.network = network
        self.agent_list = agent_list
        self.agent_num = len(agent_list)
        self.state_num = state_num
        self.action_num = action_num
        self.init_states = init_states
        self.goal_states = goal_states
        self.gamma = gamma
        self.horizon = horizon
        self.epsilon = epsilon  # The cost for waiting one unit of time

        # hash tables to store local Q functions
        # it has the form (value, precision_level)
        self.averaged_Q_table = {}

        # we build a stationary distribution table to accelerate later computation
        self.stationary_dist_table = np.zeros((self.agent_num, self.state_num, self.horizon, self.state_num))
        for i in range(self.agent_num):
            for s in range(self.state_num):
                init_dist = np.zeros(self.state_num)
                init_dist[s] = 1
                self.stationary_dist_table[i, s, :, :] = self.agent_list[i].stationary_dist(self.network, init_dist, self.horizon, True)

    def reset(self, agent_list = None):
        if agent_list is not None:
            self.agent_list = agent_list
        self.averaged_Q_table = {}
        self.stationary_dist_table = np.zeros((self.agent_num, self.state_num, self.horizon, self.state_num))
        for i in range(self.agent_num):
            for s in range(self.state_num):
                init_dist = np.zeros(self.state_num)
                init_dist[s] = 1
                self.stationary_dist_table[i, s, :, :] = self.agent_list[i].stationary_dist(self.network, init_dist,
                                                                                            self.horizon, True)

    # here, precision_level is a non-negative integer.
    # We roll out the Q at least "precision_level" times to compute it.
    def averaged_Q(self, index, global_state, local_action):
        global_state_code = global_state_encoder(global_state, self.state_num)
        Q_val = self.averaged_Q_table.get((index, global_state_code, local_action))
        if Q_val is not None:
            return Q_val

        if global_state[index] == self.goal_states[index]:
            return 0

        # if the value is not stored in the table or the precision is not enough, we need to compute it
        # first handle the first fixed local action
        result = - self.epsilon
        discount_factor = 1.0

        next_local_state = self.network.get_neighbor(global_state[index], local_action)
        if next_local_state != global_state[index]:
            # we need to check who else passed through the same edge
            for i in range(self.agent_num):
                if i != index and global_state[i] == global_state[index]:
                    p = self.stationary_dist_table[i, global_state[index], 1, next_local_state]
                    result -= (discount_factor * p)

        for t in range(1, self.horizon):
            discount_factor *= self.gamma
            for s in range(self.state_num):
                if s == self.goal_states[index]:
                    continue
                # the probability "index" is at s
                p_s = self.stationary_dist_table[index, next_local_state, t-1, s]
                # first subtract the cost for time elapse
                result -= (discount_factor * p_s * self.epsilon)
                for next_s in range(self.state_num):
                    if next_s == s:
                        continue
                    # the probability "index" goes to next_s from s
                    p_next_s = self.stationary_dist_table[index, s, 1, next_s]
                    for i in range(self.agent_num):
                        p1 = self.stationary_dist_table[i, global_state[i], t, s]
                        p2 = self.stationary_dist_table[i, s, 1, next_s]
                        if i != index:
                            result -= (discount_factor * p_s * p_next_s * p1 * p2)
                        else:
                            result -= (discount_factor * p_s * p_next_s)
        self.averaged_Q_table[(index, global_state_code, local_action)] = result
        return result

    # this function evaluates the exact local objective value under the current policy
    def local_objective(self, index, global_state):
        result = 0
        s = global_state[index]
        params = self.agent_list[index].invariant_policy[s, :]
        prob_vec = special.softmax(params)
        for a in range(self.action_num):
            result += (prob_vec[a] * self.averaged_Q(index, global_state, a-1))
        return result

    # this function computes the exact local policy gradient under the current policy
    def local_gradient(self):
        gradient = np.zeros((self.agent_num, self.state_num, self.action_num))

        # we enumerate over all possible global states
        code_max = self.state_num ** self.agent_num
        for code in range(code_max):
            global_state = global_state_decoder(code, self.state_num, self.agent_num)
            p1 = 0
            discount_factor = 1.0
            for t in range(self.horizon):
                tmp = 1.0
                for i in range(self.agent_num):
                    v = self.stationary_dist_table[i, self.init_states[i], t, global_state[i]]
                    if v < 1e-7:
                        tmp = 0
                        break
                    tmp *= v
                p1 += (discount_factor * tmp)
                discount_factor *= self.gamma
            for i in range(self.agent_num):
                for a in range(self.action_num):
                    params = self.agent_list[i].invariant_policy[global_state[i], :]
                    prob_vec = special.softmax(params)
                    term1 = np.zeros(self.action_num)
                    term1[a] = 1.0
                    term1 -= prob_vec
                    term2 = self.averaged_Q(i, global_state, a-1)
                    gradient[i, global_state[i], :] += (p1 * prob_vec[a] * term1 * term2)
        return gradient

    def mc_local_gradient(self, sample_num=100, agent_index=None):
        gradient = np.zeros((self.agent_num, self.state_num, self.action_num))
        if agent_index is not None:
            gradient = np.zeros((self.state_num, self.action_num))

        for s in range(sample_num):
            for t in range(self.horizon):
                flag = np.random.binomial(1, 1 - self.gamma)
                if flag == 1 or t == self.horizon-1:
                    # first sample a global state
                    global_state = np.zeros(self.agent_num, dtype = int)
                    for i in range(self.agent_num):
                        global_state[i] = np.random.choice(a=self.state_num, p=self.stationary_dist_table[i, self.init_states[i], t, :])
                        #print(global_state[i])

                    if agent_index is None:
                        # now we compute the gradient term at this global state
                        for i in range(self.agent_num):
                            for a in range(self.action_num):
                                params = self.agent_list[i].invariant_policy[global_state[i], :]
                                prob_vec = special.softmax(params)
                                term1 = np.zeros(self.action_num)
                                term1[a] = 1.0
                                term1 -= prob_vec
                                term2 = self.averaged_Q(i, global_state, a - 1)
                                gradient[i, global_state[i], :] += (prob_vec[a] * term1 * term2)/(1 - self.gamma)
                    else:
                        for a in range(self.action_num):
                            params = self.agent_list[agent_index].invariant_policy[global_state[agent_index], :]
                            prob_vec = special.softmax(params)
                            term1 = np.zeros(self.action_num)
                            term1[a] = 1.0
                            term1 -= prob_vec
                            term2 = self.averaged_Q(agent_index, global_state, a - 1)
                            gradient[global_state[agent_index], :] += (prob_vec[a] * term1 * term2) / (1 - self.gamma)

                    break

        return gradient/sample_num

    # We implement the local regret by local policy gradient.
    # The gradient is provided by mc_local_gradient with sample num
    def local_regret(self, sample_num=100, lr=1e-3, acc=0.1, max_round=1000):
        self.reset()
        regret_list = []
        for i in range(self.agent_num):
            local_objective_current = self.local_objective(i, self.init_states)
            back_up_policy = self.agent_list[i].invariant_policy    # first store the original policy
            for _ in range(max_round):
                self.reset()
                local_grad = self.mc_local_gradient(sample_num, i)
                self.agent_list[i].invariant_policy += lr * local_grad
                if np.linalg.norm(local_grad) < acc:
                    break
            self.reset()
            local_objective_optimal = self.local_objective(i, self.init_states)
            gap = local_objective_optimal - local_objective_current
            if gap > 0:
                regret_list.append(gap)
            else:
                regret_list.append(0.0)
            self.agent_list[i].invariant_policy = back_up_policy
        return np.array(regret_list)


class DecentralizedOptimizer:
    def __init__(self, network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon, epsilon=0.5):
        self.network = network
        self.agent_list = agent_list
        self.agent_num = len(agent_list)
        self.state_num = state_num
        self.action_num = action_num
        self.init_states = init_states
        self.goal_states = goal_states
        self.gamma = gamma
        self.horizon = horizon
        self.epsilon = epsilon  # The cost for waiting one unit of time
        self.game_simulator = trafficEnv.TrafficGame(network=self.network, action_num=self.action_num, init_states=self.init_states, goal_states=self.goal_states, epsilon=self.epsilon)

        # hash tables to store w functions
        self.w_table = {}
        self.zeta_table = {}
        self.observation_list = self.construct_obervation_table(hop=1)

    def construct_obervation_table(self, hop):
        observation_list = []
        for i in range(self.agent_num):
            local_list = []
            for j in range(self.agent_num):
                if abs(self.init_states[i] - self.init_states[j]) <= hop:
                    local_list.append(j)
            observation_list.append(local_list)
        return observation_list

    # simulate the trajectory for one epsiode, and update the local Q functions
    # rate_w and rate_zeta are the learning rate of weights and eligibility vectors
    def episode(self, rate_w, rate_zeta=0):
        self.game_simulator.reset()

        # run an episode and record the trajectory
        for t in range(self.horizon):
            global_action = np.zeros(self.agent_num, dtype=int)
            for i in range(self.agent_num):
                global_action[i] = self.agent_list[i].sample_action((-1, self.game_simulator.global_state[i]))
            self.game_simulator.step(global_action)

        # update the local Q functions
        for i in range(self.agent_num):
            for t in range(self.horizon-1):
                # first construct the feature vectors
                phi_1 = np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num)
                phi_2 = np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num)
                for c in range(len(self.observation_list[i])):
                    j = self.observation_list[i][c]
                    local_state = self.game_simulator.global_state_history[t][j]
                    phi_1[c * self.state_num + local_state] = 1.0
                    local_state = self.game_simulator.global_state_history[t+1][j]
                    phi_2[c * self.state_num + local_state] = 1.0
                local_action = self.game_simulator.global_action_history[t][i]
                phi_1[len(self.observation_list[i]) * self.state_num + local_action + 1] = 1.0
                local_action = self.game_simulator.global_action_history[t+1][i]
                phi_2[len(self.observation_list[i]) * self.state_num + local_action + 1] = 1.0

                if t == 0:
                    self.zeta_table[i] = phi_1

                w = self.w_table.get(i, np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num))
                TD_err = np.dot(phi_1, w) - self.game_simulator.global_reward_history[t][i] - np.dot(phi_2, w)
                zeta = self.zeta_table.get(i)
                self.w_table[i] = w - rate_w * TD_err * zeta
                self.zeta_table[i] = self.gamma * rate_zeta * zeta + phi_2

    def update_params(self, rate_theta):
        for i in range(self.agent_num):
            local_grad = np.zeros((self.state_num, self.action_num))
            discount_factor = 1.0
            for t in range(self.horizon):
                # first compute the Q function value
                phi = np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num)
                for c in range(len(self.observation_list[i])):
                    j = self.observation_list[i][c]
                    local_state = self.game_simulator.global_state_history[t][j]
                    phi[c * self.state_num + local_state] = 1.0
                local_action = self.game_simulator.global_action_history[t][i]
                phi[len(self.observation_list[i]) * self.state_num + local_action + 1] = 1.0
                w = self.w_table.get(i, np.zeros(len(self.observation_list[i]) * self.state_num + self.action_num))
                Q_val = np.dot(phi, w)
                local_state = self.game_simulator.global_state_history[t][i]
                params = self.agent_list[i].invariant_policy[local_state,:]
                prob_vec = special.softmax(params)
                term1 = np.zeros(self.action_num)
                term1[local_action+1] = 1.0
                term1 -= prob_vec
                self.agent_list[i].invariant_policy[local_state,:] += (rate_theta * discount_factor * Q_val * term1)
                discount_factor *= self.gamma









if __name__ == '__main__':
    np.random.seed(1)
    state_num = 2
    action_num = 2
    agent_num = 6
    horizon = 10
    gamma = 0.9
    init_states = np.zeros(agent_num, dtype=int)
    goal_states = np.ones(agent_num, dtype=int)
    learning_rate = 1e-2

    global_state = np.zeros(agent_num, dtype=int)
    network = trafficEnv.TrafficNetwork(node_num=2)
    network.add_edge((0, 1))

    agent_list = []
    for i in range(agent_num):
        agent_list.append(trafficAgent.TrafficAgent(state_num, action_num, horizon, random_init=True))
    optimizer = CentralizedOptimizer(network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon)
    print(optimizer.averaged_Q(0, global_state, -1))

    sample_size = agent_num
    objective_list = []
    regret_history = []
    for j in range(sample_size):
        objective_list.append([optimizer.local_objective(j, init_states)])
    for m in trange(1001):
        gradients = optimizer.mc_local_gradient(sample_num=300)
        for i in range(agent_num):
            agent_list[i].invariant_policy += learning_rate * gradients[i, :, :]
        optimizer.reset()
        for j in range(sample_size):
            val = optimizer.local_objective(j, init_states)
            objective_list[j].append(val)

        if m%100 == 0:
            regrets = optimizer.local_regret()
            regret_history.append(regrets)


    plt.figure()
    for j in range(sample_size):
        plt.plot(objective_list[j])
    plt.savefig("one_bridge_obejtive.png")

    params = agent_list[0].invariant_policy[0, :]
    prob_vec = special.softmax(params)
    print(prob_vec)

    regret_table = np.stack(regret_history)
    plt.figure()
    for j in range(sample_size):
        plt.plot(regret_table[:, j])
    plt.yscale('log')
    plt.savefig("one_bridge_regret.png")