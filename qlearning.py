import gym
import numpy as np
from matplotlib import pyplot as plt
import random
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

obs_limit = np.array([4.8, 5, 0.5, 5])
samples = np.random.uniform(-obs_limit, obs_limit, (10, obs_limit.shape[0]))
scaler = StandardScaler()
scaler.fit(samples)
print(samples)

np.random.seed(123)

# env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 8

x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# For LunarLander, use the following values:
x_min, x_max = -1.2, 1.2
y_min, y_max = -0.3, 1.2
xdot_min, xdot_max = -2.4, 2.4
ydot_min, ydot_max = -2, 2
theta_min, theta_max = -6.28, 6.28
thetadot_min, thetadot_max = -8, 8

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 2222
initial_q = 0  # T3: Set to 50

# Create discretization grid

x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q

x_grid = np.linspace(x_min, x_max, discr)
y_grid = np.linspace(y_min, y_max, discr)
xdot_grid = np.linspace(xdot_min, xdot_max, discr)
ydot_grid = np.linspace(ydot_min, ydot_max, discr)
theta_grid = np.linspace(theta_min, theta_max, discr)
thetadot_grid = np.linspace(thetadot_min, thetadot_max, discr)
cl_grid = np.linspace(0, 1, 2)
cr_grid = np.linspace(0, 1, 2)
q_grid = np.zeros((discr, discr, discr, discr, discr, discr, 2, 2, 8)) + initial_q


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    y = find_nearest(y_grid, state[1])
    xdot = find_nearest(xdot_grid, state[0])
    ydot = find_nearest(ydot_grid, state[0])
    theta = find_nearest(theta_grid, state[0])
    thetadot = find_nearest(thetadot_grid, state[0])
    cl = find_nearest(cl_grid, state[0])
    cr = find_nearest(cr_grid, state[0])
    return x, y, xdot, ydot, theta, thetadot, cl, cr


def get_action(state, q_values, greedy=False, epsilon=0.2):
    # x, v, th, av=get_cell_index(state)
    # best_action=np.argmax(q_values[x, v, th, av])

    x, y, xdot, ydot, theta, thetadot, cl, cr = get_cell_index(state)
    best_action = np.argmax(q_values[x, y, xdot, ydot, theta, thetadot, cl, cr])
    if greedy is False:
        rand = random.random()
        if rand < epsilon / 2:
            return 1 - best_action
        else:
            return best_action
    return best_action


def update_q_value(old_state, action, new_state, reward, done, q_array):
    # Q-value update
    # old_x, old_v, old_th, old_av = get_cell_index(old_state)
    # new_x, new_v, new_th, new_av = get_cell_index(new_state)
    # old_value=q_array[old_x, old_v, old_th, old_av,action]
    # new_value=np.max(q_array[new_x, new_v, new_th, new_av])
    # q_array[old_x, old_v, old_th, old_av,action]=(new_value*gamma+reward-old_value)*alpha+old_value
    x, y, xdot, ydot, theta, thetadot, cl, cr = get_cell_index(old_state)
    x_, y_, xdot_, ydot_, theta_, thetadot_, cl_, cr_ = get_cell_index(new_state)
    old_value = q_array[x, y, xdot, ydot, theta, thetadot, cl, cr]
    new_value = q_array[x_, y_, xdot_, ydot_, theta_, thetadot_, cl_, cr_]
    q_array[x, y, xdot, ydot, theta, thetadot, cl, cr] = (new_value * gamma + reward - old_value) * alpha + old_value
    return q_array


# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes + test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = a / (a + ep)
    while not done:
        action = get_action(state, q_grid, greedy=test, epsilon=epsilon)
        new_state, reward, done, _ = env.step(action)
        if not test:
            q_grid = update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            env.render()
        state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep - 500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep - 200):])))
# Save the Q-value array
# np.save("q_values.npy", q_grid)

# Calculate the value function

values = np.zeros(q_grid.shape[:-1])
for x in range(discr):
    for v in range(discr):
        for th in range(discr):
            for av in range(discr):
                values[x, v, th, av] = np.max(q_grid[x, v, th, av])
np.save("value_func.npy", values)

# Plot the heatmap

values = np.load('value_func_GLIE.npy')
value_map = np.zeros((discr, discr))
for x in range(discr):
    for th in range(discr):
        value_map[x, th] = np.mean(values[x, :, th, :])
plt.imshow(value_map)

# Plot the heatmap here using Seaborn or Matplotlib

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()
