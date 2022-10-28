#%%
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Read input
input_t1 = loadmat('../inputs/target_1.mat')
t1 = [[element for element in upperElement] for upperElement in input_t1['target']]
t1_data = list(zip(t1[0], t1[1]))
columns = ['x', 'y']
df1 = pd.DataFrame(t1_data, columns=columns)

input_t2 = loadmat('../inputs/target_2.mat')
t2 = [[element for element in upperElement] for upperElement in input_t2['target']]
t2_data = list(zip(t2[0], t2[1]))
df2 = pd.DataFrame(t2_data, columns=columns)

# Problem data.
i = 6-1   # choose (i=0 -> instance=1)
T = 60
x_init = np.array([1, 1, 0, 0])
A = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 0.8, 0], [0, 0, 0, 0.8]])
B = np.array([[0, 0], [0, 0], [0.2, 0], [0, 0.2]])
E = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

p1 = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
p2 = np.array([1, 0.8, 0.6, 0.4, 0.2, 0])
q1 = np.array(t1_data)
q2 = np.array(t2_data)

# Construct the problem.
X = cp.Variable((T,4)) # Matrix Tx4
U = cp.Variable((T-1,2))

te1 = 0
te2 = 0
ce = 0
constraints = [X[0] == x_init]
for t in range(T-1):
    te1 += cp.norm(E @ X[t] - q1[t], "inf")
    te2 += cp.norm(E @ X[t] - q2[t], "inf")
    ce += cp.sum_squares(U[t])
    constraints += [X[t+1] == A @ X[t] + B @ U[t]]

te1 += cp.norm(E @ X[T-1] - q1[T-1], "inf")
te2 += cp.norm(E @ X[T-1] - q2[T-1], "inf")

objective = cp.Minimize(p1[i]*te1 + p2[i]*te2 + 0.5*ce)

prob = cp.Problem(objective, constraints)

# The optimal objective value is returned by `prob.solve()`.
result = prob.solve()
# The optimal value for x is stored in `x.value`.
dfx = pd.DataFrame(X.value[:,:2], columns=columns)

# The optimal Lagrange multiplier for a constraint is stored in
# `constraint.dual_value`.
# print(constraints[0].dual_value)

# Plots

plt.figure(figsize=(46.82 * .5**(.5 * 6), 33.11 * .5**(.5 * 6))) # Magic image size line

it = 1
for frame in [df1, df2, dfx]:
    if it == 1:
        lbl = "Target 1"
        col = "red"
    elif it == 2:
        lbl = "Target 2"
        col = "magenta"
    else:
        lbl = "Tracker"
        col = "black"
    plt.plot(frame['x'], frame['y'], label=lbl, color=col)
    it += 1

plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
plt.grid(True)
plt.legend()
plt.show()
# %%
