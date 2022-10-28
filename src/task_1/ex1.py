#%%
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


T = 60
A = np.array([ [1, 0, 0.2, 0], [0, 1, 0, 0.2], [0, 0, 0.8, 0], [0, 0, 0, 0.8] ])
B = np.array([ [0, 0], [0, 0], [0.2, 0], [0, 0.2] ])
x_0 = np.array([1, 1, 0, 0])
E = np.array([ [1, 0, 0, 0], [0, 1, 0, 0] ])
y = 0.001

x = cp.Variable((T, 4))
u = cp.Variable((T-1, 2))

TE = 0
CE = 0
constr = [x[0] == x_0]
q1 = np.array(t1_data)

for t in range(T-1):
    TE += cp.norm(E @ x[t] - q1[t], 'inf')
    CE += cp.sum_squares(u[t])
    constr += [x[t + 1] == A @ x[t] + B @ u[t]]

TE += cp.norm(E @ x[T-1] - q1[T-1], "inf")
objective = cp.Minimize(TE + y*CE)
prob = cp.Problem(objective, constr)

result = prob.solve()

dfx = pd.DataFrame(x.value[:,:2], columns=columns)

print(CE.value)
print(TE.value)

"""
# Plots

plt.figure(figsize=(46.82 * .5**(.5 * 6), 33.11 * .5**(.5 * 6))) # Magic image size line

it = 1
for frame in [df1, dfx]:
    if it == 1:
        lbl = "Target 1"
        col = "red"
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

########################################################################

x = np.arange(len(labels))  # the label locations
width = 0.45 # the width of the bars

fig, ax = plt.subplots()
ax.bar(x - width/2, CE, width, label='CE')
ax.bar(x + width/2, TE, width, label='TE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Value')
ax.set_xlabel('Instance')
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()

plt.show()
"""

labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
CE = [2.981210329541508,
     5.388526746243271,
     9.233349250500275,
     12.005448231376263,
     29.339285525446343,
     43.2529144913541,
     128.6437015896907,
     203.98955726016038,
     582.9180299652983]
TE = [41.408539592113364,
     23.366679214142316,
     13.307673812335823,
     11.396413931910457,
     7.622319014254979,
     6.6565729979752,
     4.8544583139074495,
     4.344685645983705,
     3.549305998679128]

plt.plot(TE, CE, 'ro', label=labels)
plt.axis([0, 45, 0, 600])

for i in range(9):
    if i not in (2,3):
        plt.text(TE[i]+0.5,CE[i]+2,"λ=%d"%(i+1))
    elif i == 2:
        plt.text(TE[i]+0.5,CE[i],"λ=%d"%(i+1))
    else:
        plt.text(TE[i]-0.5,CE[i]+10,"λ=%d"%(i+1))

plt.grid(True)
plt.show()

# %%
