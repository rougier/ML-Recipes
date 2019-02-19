# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier
# Released under a BSD two-clauses license
#
# References: Sutton, Richard S., Barto, Andrew G. "Reinforcement Learning: An
#             Introduction", MIT Press, Cambridge, MA (1998).
# -----------------------------------------------------------------------------
import numpy as np

class EpsilonGreedy(object):
    def __init__(self, n_arms=2, epsilon=0.1, gamma=0.99):
        self.n_arms = n_arms
        self.pulls = np.zeros(n_arms, dtype=int)
        self.rewards = np.zeros(n_arms, dtype=int)
        self.epsilon = epsilon
        self.gamma = gamma
        self._epsilon = self.epsilon

        # Whether to test each arm once before actually choosing one
        self.coldstart = False

    def reset(self):
        self.pulls[...] = 0
        self.rewards[...] = 0
        self._epsilon = self.epsilon

    def update(self, arm, reward):
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self._epsilon *= self.gamma
            
    def choice(self):
        if self.coldstart and 0 in self.pulls:
            arm = np.argmin(self.pulls)
        elif np.random.uniform(0,1) < self._epsilon:
            arm = np.random.randint(self.n_arms)
        else:
            payouts = self.rewards / np.maximum(self.pulls,1)
            arm = np.argmax(payouts)
        return arm

    def __repr__(self):
        return "Epsilon Greedy (ε={}, γ={})".format(self.epsilon, self.gamma)




# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Arms probabilities
    P = [0.2, 0.8, 0.5]
    n_arms = len(P)
    
    # Number of independent runs (for averaging)
    n_runs = 1000

    # Number of consecutive trials in one run
    n_trials = 100

    R1 = np.zeros((n_runs, n_trials)) # Rewards for the player
    R2 = np.zeros((n_runs, n_trials)) # Rewards for the oracle
    
    for run in range(n_runs):
        player = EpsilonGreedy(n_arms)
        for trial in range(n_trials):
            # Player
            arm = player.choice()
            reward = np.random.uniform(0,1) < P[arm]
            R1[run,trial] = reward
            player.update(arm, reward)
            # Oracle
            R2[run,trial] = np.random.uniform(0,1) < P[np.argmax(P)]

    print("Arms: {}".format(P))
    print("Player: {}".format(player))
    print("Simulation: {} independent runs of {} consecutive trials"
          .format(n_runs, n_trials))
    print("Player reward: {:.3f} +/- {:.3f} (SE)".format(np.mean(R1), np.var(R1)))
    print("Oracle reward: {:.3f} +/- {:.3f} (SE)".format(np.mean(R2), np.var(R2)))

