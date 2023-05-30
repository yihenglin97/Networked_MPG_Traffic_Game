import numpy as np


class TrafficNetwork:
    def __init__(self, node_num):
        self.node_num = node_num
        self.adjacency_matrix = np.eye(node_num, dtype=int)
        self.adjacency_list = []
        for i in range(self.node_num):
            self.adjacency_list.append([])

    def add_edge(self, pair):
        start, end = pair
        self.adjacency_matrix[start, end] = 1
        self.adjacency_list[start].append(end)

    def get_neighbor(self, start, action_index):
        if action_index == -1:
            return start
        elif action_index >= len(self.adjacency_list[start]):
            return start
        else:
            return self.adjacency_list[start][action_index]

    def is_neighbor(self, pair):
        start, end = pair
        if self.adjacency_matrix[start, end] == 1:
            return True
        else:
            return False

class TrafficGame:
    def __init__(self, network, action_num, init_states, goal_states, epsilon=0.5):
        self.traffic_network = network
        self.init_states = init_states
        self.agent_num = init_states.shape[0]
        self.state_num = network.node_num
        self.action_num = action_num
        self.goal_states = goal_states
        self.epsilon = epsilon

        self.global_state = self.init_states
        self.global_state_history = [self.init_states]
        self.global_action = None
        self.global_action_history = []
        self.global_reward = None
        self.global_reward_history = []

        self.time_counter = 0

    def reset(self):
        self.global_state = self.init_states
        self.global_state_history = [self.init_states]
        self.global_action = None
        self.global_action_history = []
        self.global_reward = None
        self.global_reward_history = []
        self.time_counter = 0

    def step(self, actions):
        # update the global actions
        self.global_action = actions
        self.global_action_history.append(actions)

        # count the number of agents on every edge to compute the rewards
        new_global_state = np.zeros(self.agent_num, dtype=int)
        edge_count = np.zeros((self.state_num, self.state_num))
        for i in range(self.agent_num):
            current_local_state = self.global_state[i]
            # has the agent finished her trip?
            new_local_state = -1
            if current_local_state != self.goal_states[i] and actions[i] < self.action_num-1:
                new_local_state = self.traffic_network.get_neighbor(current_local_state, actions[i])
            else:
                new_local_state = current_local_state
            # record the local states globally and count the agents on edges
            new_global_state[i] = new_local_state
            edge_count[current_local_state, new_local_state] += 1

        # update the global reward
        new_global_rewards = np.zeros(self.agent_num)
        for i in range(self.agent_num):
            # if the agent has not arrived at its goal, it pays 1 unit of cost plus the congestion cost
            if self.goal_states[i] != self.global_state[i]:
                new_global_rewards[i] -= self.epsilon
                if self.global_state[i] != new_global_state[i]:
                    new_global_rewards[i] -= edge_count[self.global_state[i], new_global_state[i]]
        self.global_reward = new_global_rewards
        self.global_reward_history.append(new_global_rewards)

        # update the global state and see whether the game is finished
        self.global_state = new_global_state
        self.global_state_history.append(self.global_state)
        self.time_counter += 1
        num_unfinished = 0
        for i in range(self.agent_num):
            if self.global_state[i] != self.goal_states[i]:
                num_unfinished += 1
        return self.global_reward, num_unfinished


if __name__ == '__main__':
    network = TrafficNetwork(node_num=2)
    network.add_edge((0, 1))
    agent_num = 5
    action_num = 2
    init_states = np.zeros(agent_num, dtype=int)
    goal_states = np.ones(agent_num, dtype=int)
    game = TrafficGame(network, action_num, init_states, goal_states)

    _, unfinished = game.step(-np.ones(agent_num, dtype=int))
    print("States: {} \\Reward: {}\\Unfinished: {}".format(game.global_state, game.global_reward, unfinished))

    _, unfinished = game.step(np.zeros(agent_num, dtype=int))
    print("States: {} \\Reward: {}\\Unfinished: {}".format(game.global_state, game.global_reward, unfinished))



