""""""
"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: nriojas3 (replace with your User ID)
GT ID: 903646605 (replace with your GT ID)
"""

import random as rand
import numpy as np


class QLearner(object):
    """
    This is a Q learner object.
    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available..
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def __init__(
            self,
            num_states=100,
            num_actions=4,
            alpha=0.2,
            gamma=0.9,
            rar=0.5,
            radr=0.99,
            dyna=0,
            verbose=False,
    ):
        """
        Constructor method
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.empty = -1
        # initialize Q and R in the same way
        self.Q, self.R = np.zeros((num_states, num_actions)), np.zeros((num_states, num_actions))
        # initialize T with nans to house experience tuple T'[s,a,s']
        # note that 2 extra points are reserved to hold s_prime with the largest value and its value
        self.T = np.full((num_states, num_actions, num_states + 2), self.empty)

    def author(self):
        return "nriojas3"

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table
        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """
        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        # if random action isn't chosen, look through Q table at row s and choose highest key value
        else:
            action = np.argmax(self.Q[s])
        self.s, self.a = s, action
        return self.a

    def update_Q(self, s, a, s_prime, r):
        # implement the equation for computing Q given experience tuple <s,a,s',r>
        # Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmaxa'(Q[s', a'])]) -> 3-06 Lecture 4
        Q_previous = self.Q[s][a]
        current_s_argmax = np.argmax(self.Q[s_prime])
        Q_current = self.Q[s_prime][current_s_argmax]
        self.Q[s][a] = ((1 - self.alpha) * Q_previous) + (self.alpha * (r + (self.gamma * Q_current)))

    def update_R(self, s, a, r):
        #  implement the equation for computing R
        # R'[s, a] = (1 - α) · R[s, a] + α · r -> 3-07 Lecture 5
        self.R[s][a] = ((1 - self.alpha) * self.R[s][a]) + (self.alpha * r)

    def update_T_max_s_prime(self, s, a, s_prime, val):
        # second to last position in the T matrix reserved for max s_prime
        self.T[s][a][-2] = s_prime
        # last position in T matrix reserved for s_prime value
        self.T[s][a][-1] = val

    def query(self, s_prime, r):
        """
        Update the Q table and return an action
        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """
        # update Q Table ------------------------------------------------------------------------------------------
        self.update_Q(self.s, self.a, s_prime, r)

        # implement Dyna --------------------------------------------------------------------------------------------
        if self.dyna:
            # UPDATE MODEL
            # T'[s,a,s'] update
            if self.T[self.s][self.a][s_prime] == self.empty:
                self.T[self.s][self.a][s_prime] = 1
                # check if a largest s_prime for state s and action a have not been set
                if self.T[self.s][self.a][-1] == self.empty:
                    self.update_T_max_s_prime(self.s, self.a, s_prime, 1)
            else:
                self.T[self.s][self.a][s_prime] += 1
                # check if the updated s_prime frequency is greater than the current max
                if self.T[self.s][self.a][s_prime] > self.T[self.s][self.a][-1]:
                    self.update_T_max_s_prime(self.s, self.a, s_prime, self.T[self.s][self.a][s_prime])

            # R'[s,a] update
            self.update_R(self.s, self.a, r)

            # Dyna Q
            for i in range(self.dyna):
                # HALLUCINATE
                # s = random, a = random
                s = rand.randint(0, self.num_states - 1)
                a = rand.randint(0, self.num_actions - 1)
                # s' infer from T[]
                if self.T[s][a][-1] != self.empty:
                    # this setup eliminates using np.argmax to always look up max values at O(num_states)
                    s_prime_hallucinate = self.T[s][a][-2]
                    # r = R[s,a]
                    r_hallucinate = self.R[s][a]
                    # Q update
                    # update Q with experience tuple <s,a,s',r>
                    self.update_Q(s, a, s_prime_hallucinate, r_hallucinate)

        # decide on action to take ---------------------------------------------------------------------------------
        # if random action is chosen, perform random action
        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        # if random action isn't chosen, look through Q table at row s_prime and choose highest key value
        else:
            action = np.argmax(self.Q[s_prime])

        # update state and action ----------------------------------------------------------------------------------
        self.s = s_prime
        self.a = action

        # update rar with rar = rar * radr -------------------------------------------------------------------------
        self.rar *= self.radr

        return action


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
