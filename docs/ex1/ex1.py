import cvxpy as cp
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd




input_t1 = loadmat('../inputs/target_1.mat')
t1 = [[element for element in upperElement] for upperElement in input_t1['target']]
t1_data = list(zip(t1[0], t1[1]))
columns = ['x', 'y']
df1 = pd.DataFrame(t1_data, columns=columns)


T = 50
A = np.array([ [1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 0.8, 0], [0, 0, 0, 0.8] ])
B = np.array([ [0, 0], [0, 0], [0.2, 0], [0, 0.2] ])
x_0 = np.array([1, 1, 0, 0])
E = np.array([ [1, 0, 0, 0], [0, 1, 0, 0] ])
u = np.array([1, 1])
y = 0.01
x = cp.Variable((T, 4))
u = cp.Variable((T-1, 2))
cost = 0
TE = 0
CE = 0
constr = [x[0] == x_0]
q1 = np.array(t1_data)

for t in range(T-1):
    TE += cp.norm(E @ x[t] - q1[t], 'inf')
    CE += y * cp.sum_squares(u[t])
    cost = TE + CE
    constr += [x[t + 1] == A @ x[t] + B @ u[t]]

TE += cp.norm(E @ x[T-1] - q1[T-1], "inf")
objective = cp.Minimize(TE + y*CE)
prob = cp.Problem(objective, constr)

result = prob.solve()

dfx = pd.DataFrame(x.value[:,:2], columns=columns)

print(CE.value)
print(TE.value)

# Plots

# plt.figure(figsize=(46.82 * .5**(.5 * 6), 33.11 * .5**(.5 * 6))) # Magic image size line

# it = 1
# for frame in [df1, dfx]:
#     if it == 1:
#         lbl = "Target 1"
#         col = "red"
#     else:
#         lbl = "Tracker"
#         col = "black"
#     plt.plot(frame['x'], frame['y'], label=lbl, color=col)
#     it += 1

# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
# plt.grid(True)
# plt.legend()
# plt.show()

labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
CE = [0.227, 1.488, 9.215, 8.639, 12.723, 16.016, 24.931, 12.465, 2.493]
TE = [80.566, 69.326, 13.175, 9.421, 4.854, 3.943, 2.9, 2.9, 2.9]

x = np.arange(len(labels))  # the label locations
width = 0.45 # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, CE, width, label='CE')
rects2 = ax.bar(x + width/2, TE, width, label='TE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Value')
ax.set_xlabel('Instance')
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()