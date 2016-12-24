#encoding: utf-8


class DecisionPolicy:
    '''
    decision policy abstract class
    '''
    def select_action(self, current_state, step):
        pass

    def update_q(self, state, action, reward, next_state):
        pass