# encoding: utf-8
import numpy as np
import environment
import util
from decision_rand import RandomDecisionPolicy
from decision_ql import QLearningDecisionPolicy
from datetime import datetime as dt

def run_simu(policy, initial_budget, initial_num_stocks, prices, hist, debug=False):
    # initializatin
    budget = initial_budget
    num_stocks = initial_num_stocks
    share_value = 0

    transition = list()

    for i in xrange(len(prices) - hist - 1):
        if i % 100 == 0 and debug:
            print('progress {:.2f}%'.format(float(100 * i) / (len(prices) - hist - 1)))
        # the state is a hist+2 dim vector
        current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks)))

        # calc current protfolio value
        current_portfolio = budget + num_stocks * share_value

        # select action
        action = policy.select_action(current_state, i)

        # share value
        share_value = float(prices[i + hist + 1])

        # update portfolio values based on action
        if action == "Buy" and budget >= share_value:
            budget -= share_value
            num_stocks += 1
        elif action == "Sell" and num_stocks > 0:
            budget += share_value
            num_stocks -= 1
        else:
            action = "Hold"

        # calc new portofolio after tacking action
        new_portfolio = budget + num_stocks * share_value

        # calc reward from tacking an action at a state
        reward = new_portfolio - current_portfolio

        next_state = np.asmatrix(np.hstack((prices[i+1:i+hist+1], budget, num_stocks)))

        transition.append((current_state, action, reward, next_state))

        # update the policy after experiencing a new action
        policy.update_q(current_state, action, reward, next_state)

    # compute final portfolio worth
    portfolio = budget + num_stocks * share_value

    if debug:
        print('${} budget\t{} shares'.format(budget, num_stocks))

    return portfolio


def run_simus(policy, budget, num_stocks, prices, hist, num_tries):
    final_portofolios = list()

    for i in xrange(num_tries):
        print("simuration no.%d" % (i))
        final_portofolio = run_simu(policy, budget, num_stocks, prices, hist, False)
        final_portofolios.append(final_portofolio)
    avg, std = np.mean(final_portofolios), np.std(final_portofolios)
    return avg, std


if __name__ == '__main__':
    # simuration date
    tstr = dt.now().strftime('%Y%m%d-%H%M%S')
    print("%s [training] start" % tstr)

    tstr = dt.now().strftime('%Y%m%d-%H%M%S')
    print("%s [loading data] start" % tstr)

    prices = environment.get_prices('MSFT', '1992-07-22', '2016-07-22')
    util.plot_prices(prices)

    tstr = dt.now().strftime('%Y%m%d-%H%M%S')
    print("%s [loading data] finish" % tstr)

    actions = ["Buy", "Sell", "Hold"]
    hist = 200
    #policy = RandomDecisionPolicy(actions)
    policy = QLearningDecisionPolicy(actions, hist+2, "train20161226-002008")

    budget = 1000.0
    num_stocks = 0
    num_tries = 10
    avg, std = run_simus(policy, budget, num_stocks, prices, hist, num_tries)
    policy.save_model("train%s" % tstr, num_tries)

    tstr = dt.now().strftime('%Y%m%d-%H%M%S')
    print("%s portfolio avg: $%f, std: %f" % (tstr, avg, std))

    tstr = dt.now().strftime('%Y%m%d-%H%M%S')
    print("%s [training] finish" % (tstr))