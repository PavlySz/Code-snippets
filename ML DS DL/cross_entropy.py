'''
Compute Cross-Entropy loss function
Sauce: Deep Learning Specialization by Prof. Andre Ng - Coursera
'''
import numpy as np


def compute_cost(A2, Y):
    '''
    Computes the cross-entropy cost given in equation

    Args:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2
        [Note that the parameters argument is not used in this function,
        but the auto-grader currently expects this parameter.
        Future version of this notebook will fix both the notebook
        and the auto-grader so that `parameters` is not needed.
        For now, please include `parameters` in the function signature,
        and also when invoking this function.]

    Returns:
        cost -- cross-entropy cost given equation (13)
    '''

    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.sum(np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y))
    cost = - np.multiply(1/m, logprobs)

    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect.
                                    # E.g., turns [[17]] into 17
    assert isinstance(cost, float)

    return cost
