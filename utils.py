import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def generate_episode_from_Q(env, Q, epsilon, nA, max_steps):
    """ generates an episode from following the epsilon-greedy policy """
    episode = []
    state = env.reset()
    state = str(state[0])
    n_steps = 0
    while True:
        if state in list(Q.keys()):
            action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA))
        else:
            action = env.action_space.sample()

        # take a step in the environement
        next_state, reward, done, _, info = env.step(action) ## YOUR CODE HERE
        episode.append((str(state), action, reward))
        state = next_state
        n_steps +=1
        if done or n_steps > max_steps:
            break
    return episode

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_s = np.array([epsilon/nA for i in range(nA)])
    best_a = np.argmax(Q_s)
    policy_s[best_a] +=  1 - epsilon
    return policy_s

def update_Q(env, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    for i, state in enumerate(states):
        old_Q = Q[state][actions[i]]
        Q[state][actions[i]] = old_Q + alpha * (np.sum(discounts[:-(i+1)] * rewards[i:]) - old_Q)
    return Q

def get_state_from_str(obs: str):

    obs_splitted = obs[1:-1].split(',')

    return int(obs_splitted[0]), int(obs_splitted[1])

def plot_state_value_function(Q, model:str):
    V = dict((k, np.max(val)) for k, val in Q.items())

    x = np.array([get_state_from_str(k)[0] for k in V.keys()])
    y = np.array([get_state_from_str(k)[1] for k in V.keys()])
    z = np.array([[val] for val in V.values()])  # Remove unnecessary nested list

    X = np.unique(x)
    Y = np.unique(y)

    if model=='sarsa':
        Z = np.ones((len(Y), len(X)))
    else:
        Z = np.zeros((len(Y), len(X)))
    # fill values in state value matrix
    for idx_y, coord_x in enumerate(Y):
        for idx_x, coord_y in enumerate(X):

            if '(' + str(coord_x) + ', ' + str(coord_y) + ')' in list(V.keys()):

                Z[idx_y, idx_x] = V['(' + str(coord_x) + ', ' + str(coord_y) + ')']

    

    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(X, Y)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('V(s)')
    ax.set_title('State Value Function V(s)')

    # Show plot
    plt.show()