# Networked Markov Potential Game: Traffic Game

This repository contains the code for the numerical experiments in the manuscript "*Convergence Rates for Localized Actor-Critic in Networked Markov Potential Games*."</br>

## To reproduce the plots in the manuscript:

Run main.py. Then, run plotFigures.py. The plots will be stored in the folder "Figures".

## Code structure:

- 'trafficEnv.py': The multi-agent MDP environment of the traffic game.
- 'trafficAgent.py': Simulate the behavior of each agent.
- 'learningAgent.py': Contain decentralized algorithms for learning and centralized algorithms for evaluation.
- 'main.py': Run the experiment with multiple random seeds and record the results.
- 'plot.py': Plot the regret curves based on the recorded results.
