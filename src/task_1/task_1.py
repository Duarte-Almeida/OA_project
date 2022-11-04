#%%
import cvxpy as cp
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib

# use tex fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Computer Modern",
})

# Optimization problem constants
q1 = loadmat('../inputs/target_1.mat')["target"].T
T = q1.shape[0]
A = np.array([ [1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 0.8, 0], [0, 0, 0, 0.8] ])
B = np.array([ [0, 0], [0, 0], [0.2, 0], [0, 0.2] ])
x_0 = np.array([1, 1, 0, 0])
E = np.array([ [1, 0, 0, 0], [0, 1, 0, 0] ])
lambs = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

# Optimization problem variables
x = cp.Variable((T, 4))
u = cp.Variable((T - 1, 2))

# Optimization problem constraints
TE = 0
CE = 0
constr = [x[0] == x_0]

for t in range(T - 1):
    TE += cp.norm(E @ x[t] - q1[t], 'inf')
    CE += cp.sum_squares(u[t])
    constr += [x[t + 1] == A @ x[t] + B @ u[t]]

TE += cp.norm(E @ x[T - 1] - q1[T - 1], "inf")

# Record values for TE and CE for each lambda parameter value
TEs = []
CEs = []

# Solve problem for each lambda value and plot obtained trajectory
for i in range(len(lambs)):
    objective = cp.Minimize(TE + lambs[i] * CE)
    prob = cp.Problem(objective, constr)
    result = prob.solve()
    TEs.append(TE.value)
    CEs.append(CE.value)

    # Magic image size line
    plt.figure(figsize=(46.82 * .5**(.5 * 6), 33.11 * .5**(.5 * 6))) 

    plt.plot(x[:, 0].value, x[:, 1].value, label = "Vehicle", color = "black", marker = 'o',linewidth = 1, markersize = 1.5)
    plt.plot(q1[:, 0], q1[:, 1], label = "Target 1", color = "red", marker='o', linewidth = 1, markersize = 1.5)
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.minorticks_on()
    plt.grid(which = "major", linestyle = "-", alpha = 0.6)
    plt.grid(which = "minor", linestyle = "--", alpha = 0.4)
    plt.tick_params(which = "minor", width = 0)
    plt.tick_params(which = "major", direction = "in")
    plt.text(x_0[0] + .05, x_0[1] + .1, "Vehicle", {"color": "black"})
    plt.text(q1[0][0] + .05, q1[0][1] - .15, "Target", {"color": "red"})
    plt.title(f"$\lambda = {lambs[i]}$")
    plt.savefig(f"./output/ex_1_i={i + 1}.pdf")
    plt.cla()

plt.plot(TEs, CEs, 'ro', label = range(i, len(TEs) + 1), color = "blue", markersize = 2)
plt.axis([0, 45, 0, 600])
plt.xlabel("TE")
plt.ylabel("CE")

# fine tune where labels should be placed for each (TE, CE) pair
offsets = [[0.2, 10],  # 1
           [0.2, 10],  # 2
           [0.5, 10],  # 3
           [0.5, 10],  # 4
           [0.5, 2],   # 5
           [0.2, 10],  # 6
           [0.5, 1],   # 7
           [0.5, 2],   # 8
           [0.8, -10]] # 9

for i in range(len(TEs)):
    plt.text(TEs[i] + offsets[i][0], 
             CEs[i] + offsets[i][1], f"$\lambda_{i + 1}$")
    
plt.minorticks_on()
plt.grid(which = "major", linestyle = "-", alpha = 0.6)
plt.grid(which = "minor", linestyle = "--", alpha = 0.4)
plt.tick_params(which = "minor", width = 0)
plt.tick_params(which = "major", direction = "in")
plt.savefig("./output/TEvsCE.pdf")

# %%
