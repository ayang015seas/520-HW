import numpy as np
import math

# save global arrays 
# backward forward reward array 
back_for = np.zeros((5, 2))

# forward rewards


reward_forward = np.zeros((5, 5))
reward_backward = np.zeros((5, 5))

reward_forward[0,0] = 3
reward_forward[1,0] = 3
reward_forward[2,0] = 3
reward_forward[3,0] = 3
reward_forward[4,0] = 3
reward_forward[4,4] = 10

reward_backward[0,0] = 3
reward_backward[1,0] = 3
reward_backward[2,0] = 3
reward_backward[3,0] = 3
reward_backward[4,0] = 3
reward_backward[4,4] = 10


forward_transition = [[{0: 0.3}, {1: 0.7}],[{0: 0.3}, {2: 0.7}],
                      [{0: 0.3}, {3: 0.7}],[{0: 0.3}, {4: 0.7}],
                      [{0: 0.3}, {4: 0.7}]]

backward_transition = [[{0: 0.7}, {1: 0.3}],[{0: 0.7}, {2: 0.3}],
                      [{0: 0.7}, {3: 0.3}],[{0: 0.7}, {4: 0.3}],
                      [{0: 0.7}, {4: 0.3}]]

def getKey(obj):
    for key in obj:
        return key
    
def value_iteration(penalty, threshold):
    max_delta = 1
    state_rewards = np.zeros(5)
#     for j in range(0, 2):
    while (max_delta > threshold):
        new_state_rewards = np.zeros(5)
        for i in range(0, len(forward_transition)):
            # now we compute forward for individual state
            f_trans = forward_transition[i]
            b_trans = backward_transition[i]
            
            f_trans_expected = 0
            for trans in f_trans:
                next_state = getKey(trans)
                print(i, "to", next_state)
                print(reward_forward[i, next_state], ",", state_rewards[next_state])
                print(trans[next_state])
                reward = (reward_forward[i, next_state] + (penalty * state_rewards[next_state]))
                f_trans_expected += (trans[next_state] * reward)

            b_trans_expected = 0
            for trans in b_trans:
                next_state = getKey(trans)
                reward = (reward_backward[i, next_state] + (penalty * state_rewards[next_state]))
                b_trans_expected += (trans[next_state] * reward)
            
            back_for[i, 0] = f_trans_expected
            back_for[i, 1] = b_trans_expected
            
            new_state_rewards[i] = max(f_trans_expected, b_trans_expected)
        # check margins
        margins = new_state_rewards - state_rewards
        max_delta = np.max(margins)
#         print(new_state_rewards)
        state_rewards = new_state_rewards
    return state_rewards
        

def find_optimal(decision_matrix):
    decisions = []
    for i in range(len(decision_matrix)):
        decisions.append(np.argmax(decision_matrix[i]))
    return decisions
                
sr = value_iteration(0.9, 0.001)
print(sr)
print(back_for)
decisions = find_optimal(back_for)
print(decisions)
print("FINISHED")
