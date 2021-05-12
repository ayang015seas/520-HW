# # Viterbi algorithm


import numpy as np


def viterbi(pi, A, phi, X):
    """ Viterbi algorithm for finding MAP estimates for 
    Hidden Markov Model

    NOTE: Feel free to modify this function

    Args:
        pi (numpy.array): Initial distribution
        1D k x 1 array 

        A (numpy.array): State transition probabilities
        k x k matrix 
        A[i,j] i->j transition (row, col)
        states are indexed from 0 
        
        phi (numpy.array): Emission probabilities
        d = categories of observations 
        d x K matrix 
        (observation, hidden state)
        observations indexed from 0 again 
        
        X (numpy.array): Observation sequence
        N x 1 vector
        
        
    Returns:
        (Z, prob)
            Z (list): MAP estimate of states
            N x 1 hidden states -> 
            prob (float): joint distribution (state, observation)
            probability of seeing hidden state and observations 

    """
    # Your code goes here
    # create 2 arrays of dimension N x k 
    # array 1 keeps track of added log probabilities 
    N = len(X)
    k = A.shape[0]
    transitions = A
    
    log_prob = np.zeros((N, k))
    categories = np.zeros((N, k))
    
    log_prob[0] = np.log(np.asarray(pi)) + np.log(phi[X[0]])
    print("initial")
    print(np.asarray(pi), phi[X[0]])
    
    # array 2 keeps track of backtrack values 
    # looping through n items 
    # for each item loop through k states 
    # be very careful 
    for i in range(1, N):
        observation = X[i]
        local_emit = phi[observation]
        local_test = []
        local_transitions = []
        for j in range(0, k):
            local_transitions.append(transitions[j])
            local_test.append(local_emit[j])
            local_transition = np.log(transitions[j]) + log_prob[i - 1]
            local_transition += np.log(np.full((k,), local_emit[j]))
#             local_test.append(local_transition)
            max_index = np.argmax(local_transition)
            log_prob[i][j] = local_transition[max_index]
            categories[i][j] = max_index
#         print("emit", local_test)
#         print("transition", local_transitions)
    
#     print(log_prob)
#     print(categories)
    states = []
    log_sum = np.max(log_prob[N - 1])
    # backtrack 
    current_state = categories[N - 1][np.argmax(log_prob[N - 1])]
    states.append(current_state)
    for i in range(N - 1, 0, -1):
#         print("Curr", current_state.item(), categories)
        current_state = categories[i][int(current_state.item())]
        states.append(current_state)
        log_sum = log_sum + log_prob[i - 1][int(current_state.item())]
    states.reverse()
    Z = states
#     print("HAPPY SAD", np.exp((log_prob[N - 1])))
    prob = np.exp(np.max(log_prob[N - 1]))
    # when backtracking at end add up all array values and log exponentiate as well 
    
    return Z, prob



# setup variables 
# happy -> 0 
# sad - > 1 
pi = np.zeros((2,))
pi[0] = 0.5
pi[1] = 0.5
transitions = np.zeros((2,2))
transitions[0][0] = (4/5)
transitions[0][1] = (1/2)
transitions[1][0] = (1/5)
transitions[1][1] = (1/2)

emissions = np.zeros((3,2))
emissions[0][0] = 5/10 
emissions[0][1] = 1/10 
emissions[1][0] = 3/10 
emissions[1][1] = 2/10 
emissions[2][0] = 2/10 
emissions[2][1] = 7/10 

observations1 = np.asarray([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
observations2 = np.asarray([0, 0, 0, 2, 2, 2, 2, 2, 2, 2])
observations3 = np.asarray([2, 1, 2, 0, 0, 0, 1, 2, 0, 0])
test1 = np.asarray([0, 2])


print(pi)
Z1, prob1 = viterbi(pi, transitions, emissions, observations1)
Z2, prob2 = viterbi(pi, transitions, emissions, observations2)
Z3, prob3 = viterbi(pi, transitions, emissions, observations3)

print(Z1)
print(prob1)
print(Z2)
print(prob2)
print(Z3)
print(prob3)
# sing -> 0 
# walk -> 1
# TV -> 3 
