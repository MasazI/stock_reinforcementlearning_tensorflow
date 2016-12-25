# encoding: utf-8
import random
from decision_inf import DecisionPolicy


class RandomDecisionPolicy(DecisionPolicy):
    '''
    random decision policy
    '''
    def __init__(self, actions):
        self.actions = actions

    def select_action(self, current_state, step):
        action = self.actions[random.randint(0, len(self.actions) - 1)]
        return action