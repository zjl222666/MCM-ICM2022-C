import csv
from itertools import count


# parameters setting

total_days = 1822
gold_ban_days = [7, 6] # 表示黄金不交易的日子


# pre从201条起一共1626条


class TradeEnv:
    def __init__(self, path_for_gold, path_for_bitcoin, pre_of_bitcoin, pre_of_gold) -> None:
        with open(path_for_gold) as f:
            self.gold_price = list(map(lambda x: float(x[2]), list(csv.reader(f))[1:]))
        with open(path_for_bitcoin) as f:
            self.bitcoin_price = list(map(lambda x: float(x[2]), list(csv.reader(f))[1:]))
        with open(pre_of_bitcoin) as f:
            self.bitcoin_pre = list(map(lambda x: float(x), f.readline().split()))
        with open(pre_of_gold) as f:
            self.gold_pre = list(map(lambda x: float(x), f.readline().split()))
        self.gold_cost = 0.02
        self.bitcoin_cost = 0.03
        self.states = [
            'cur_day',
            'cur_dollars',
            'cur_bitcoin',
            'cur_gold',
            'gold_price',
            'bitcoin_price',
            'pre_gold',
            'pre_bitcoin',
            'total_money'
        ]
        self.actions = [
            'gold_changes',
            'bitcoin_changes',
        ]
        self.reward_gamma = 0.001 

    def cal_reward(self, state, next_state):
        '''
        calculate the reward for DDPG
            - revenue based
            - predictor based
        '''
        return  ((next_state['total_money'] - state['total_money'] )) * self.reward_gamma * state['cur_day'] + self.punish

    def reset(self):
        return [0, 1000, 0, 0, self.gold_price[0], self.bitcoin_price[0],self.gold_price[0], self.bitcoin_price[0], 1000]

    def step(self, state_list, action_list):
        state = {}
        action = {}
        for i, s in enumerate(self.states):
            state[s] = state_list[i]
        
        self.punish = 0
        for i, a in enumerate(self.actions):
            action[a] = action_list[i]
        
        next_state = {}
        
        try:
            for day in gold_ban_days:
                if state['cur_day'] % day == 0:
                    if not action['gold_changes'] == 0 or not action['gold_changes'] == 0:
                        self.punish -= 10
                        action['gold_changes'] = 0

            if state['cur_gold'] + action['gold_changes'] < 0:
                action['gold_changes'] = -state['cur_gold']
                self.punish -=  10
            
            if state['cur_bitcoin'] + action['bitcoin_changes'] < 0:
                action['bitcoin_changes'] = -state['cur_bitcoin']
                self.punish -= 10
            if action['gold_changes'] * self.gold_price[state['cur_day']] + action['bitcoin_changes'] * self.bitcoin_price[state['cur_day']] > state['cur_dollars']:
                self.punish -= 20
                if action['gold_changes'] > 0  and action['bitcoin_changes'] > 0:
                    tmp_sum = action['gold_changes'] + action['bitcoin_changes']
                    action['gold_changes'] = state['cur_dollars'] / self.gold_price[state['cur_day']] * action['gold_changes'] / tmp_sum
                    action['bitcoin_changes'] = state['cur_dollars'] / self.bitcoin_price[state['cur_day']] * action['bitcoin_changes'] / tmp_sum
                elif action['gold_changes'] > 0:
                    action['gold_changes'] = state['cur_dollars'] / self.gold_price[state['cur_day']]
                else:
                    action['bitcoin_changes'] = state['cur_dollars'] / self.bitcoin_price[state['cur_day']]
            next_state['cur_day'] = state['cur_day'] + 1
            next_state['cur_dollars'] = state['cur_dollars'] 
            next_state['cur_gold'] = state['cur_gold']
            next_state['cur_bitcoin'] = state['cur_bitcoin']
            if action['bitcoin_changes'] > 0:
                next_state['cur_dollars'] -= action['bitcoin_changes'] * self.bitcoin_price[state['cur_day']]
                next_state['cur_bitcoin'] += action['bitcoin_changes'] * (1 - self.bitcoin_cost)
            else:
                next_state['cur_bitcoin'] += action['bitcoin_changes']
                next_state['cur_dollars'] -= action['bitcoin_changes'] * self.bitcoin_price[state['cur_day']] *  (1 - self.bitcoin_cost)
            
            if action['gold_changes'] > 0:
                next_state['cur_dollars'] -= action['gold_changes'] * self.gold_price[state['cur_day']]
                next_state['cur_gold'] += action['gold_changes'] * (1 - self.gold_cost)
            else:
                next_state['cur_gold'] += action['gold_changes']
                next_state['cur_dollars'] -= action['gold_changes'] * self.gold_price[state['cur_day']] *  (1 - self.gold_cost)
            
            next_state['pre_gold'] = (self.gold_pre[state['cur_day'] - 200] - self.gold_price[state['cur_day']]) / \
                            self.gold_price[state['cur_day']] if state['cur_day'] >= 200 else 0

            next_state['pre_bitcoin'] = (self.bitcoin_pre[state['cur_day'] - 200] - self.bitcoin_price[state['cur_day']]) / \
                            self.bitcoin_price[state['cur_day']] if state['cur_day'] >= 200 else 0

            next_state['gold_price'] = self.gold_price[next_state['cur_day']]
            next_state['bitcoin_price'] = self.bitcoin_price[next_state['cur_day']]
            next_state['total_money'] = (next_state['cur_gold'] * self.gold_price[next_state['cur_day']] + \
                                        next_state['cur_bitcoin'] * self.bitcoin_price[next_state['cur_day']] + \
                                        next_state['cur_dollars'])
            next_state_list = []
            for i, s in enumerate(self.states):
                next_state_list.append(next_state[s])
            return next_state_list, self.cal_reward(state, next_state), next_state['cur_day'] >= total_days
        except:
            print("!warn")
            return state_list, 0, state['cur_day'] >= total_days

def main():
    test_env = TradeEnv("GOLD.csv", "bitcoin.csv", "bitcoin_pre.txt","gold_pre.txt")

if __name__ == "__main__":
    main()


        
