# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier
# Released under a BSD two-clauses license
#
# References: Thompson, William R. "On the likelihood that one unknown
#             probability exceeds another in view of the evidence of two
#             samples". Biometrika, 25(3–4):285–294, 1933.
#             DOI: 10.2307/2332286
# -----------------------------------------------------------------------------
import numpy as np

class ThompsonSampling(object):
    """ Thompson sampling """
    
    def __init__(self, n_arms=2, alpha=1.0, beta=1.0):
        self.n_arms = n_arms
        self.pulls = np.zeros(n_arms, dtype=int)
        self.rewards = np.zeros(n_arms, dtype=int)
        self.alpha = alpha
        self.beta = beta

        # Whether to test each arm once before actually choosing one
        self.coldstart = True    

    def reset(self):
        self.pulls[...] = 0
        self.rewards[...] = 0
        
    def update(self, arm, reward):
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        
    def choice(self):
        if self.coldstart and 0 in self.pulls:
            arm = np.argmin(self.pulls)
        else:
            P = np.random.beta(self.alpha+self.rewards,
                               self.beta+self.pulls-self.rewards)
            arm = np.argmax(P)
        return arm
        
    def __repr__(self):
        return "Thompson sampling (α={}, β={})".format(self.alpha, self.beta)



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
        player = ThompsonSampling(n_arms)
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

