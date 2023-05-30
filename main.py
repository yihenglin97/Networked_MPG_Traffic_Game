import numpy as np
import trafficEnv
import math
from scipy import special
import trafficAgent
import learningAgent
from tqdm import trange
import matplotlib.pyplot as plt


def test(seed):
    np.random.seed(seed)
    state_num = 8
    action_num = 3
    agent_num = 12
    horizon = 15
    gamma = 0.9
    init_states = np.zeros(agent_num, dtype=int)
    init_states[3:6] = np.ones(3, dtype=int)
    init_states[6:9] = 2 * np.ones(3, dtype=int)
    init_states[9:12] = 3 * np.ones(3, dtype=int)
    goal_states = 7 * np.ones(agent_num, dtype=int)
    learning_rate = 1e-2
    epsilon = 0.5

    global_state = np.zeros(agent_num, dtype=int)
    network = trafficEnv.TrafficNetwork(node_num=state_num)
    network.add_edge((0, 4))
    network.add_edge((1, 4))
    network.add_edge((1, 5))
    network.add_edge((2, 5))
    network.add_edge((2, 6))
    network.add_edge((3, 6))
    network.add_edge((4, 7))
    network.add_edge((5, 7))
    network.add_edge((6, 7))

    agent_list = []
    for i in range(agent_num):
        agent_list.append(trafficAgent.TrafficAgent(state_num, action_num, horizon, random_init=True))
    optimizer = learningAgent.CentralizedOptimizer(network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon, epsilon)
    decentralized_optimizer = learningAgent.DecentralizedOptimizer(network, agent_list, state_num, action_num, init_states, goal_states, gamma, horizon, epsilon)
    print(optimizer.averaged_Q(0, global_state, -1))

    sample_size = agent_num
    objective_list = []
    regret_history = []
    for j in range(sample_size):
        objective_list.append([optimizer.local_objective(j, init_states)])
    for m in trange(4000):
        if m == 0 or m % 100 == 99:
            regrets = optimizer.local_regret(sample_num=30, lr=1e-2, acc=0.5)
            regret_history.append(regrets)

        for i in range(agent_num):
            for _ in range(10):
                decentralized_optimizer.episode(rate_w=1e-3)
            decentralized_optimizer.update_params(rate_theta=1e-3)
        optimizer.reset()
        for j in range(sample_size):
            val = optimizer.local_objective(j, init_states)
            objective_list[j].append(val)

    # save the result to an npy file for plot
    objective_record = np.array(objective_list)
    np.save("./Data/inexact_multi_bridge_objective_record_{}.npy".format(seed), objective_record)

    for i in range(agent_num):
        params = agent_list[i].invariant_policy[i // 3, :]
        prob_vec = special.softmax(params)
        print(prob_vec)

    regret_table = np.stack(regret_history)

    # save the regret of each agent to an npy file for plot
    np.save("./Data/inexact_multi_bridge_regret_table_{}.npy".format(seed), regret_table)

if __name__ == '__main__':
    for i in range(10):
        print("Test seed {}".format(i))
        test(i)