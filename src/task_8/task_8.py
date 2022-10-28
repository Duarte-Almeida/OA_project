import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib

# column vector of instants of measurements
t_i = 0
t_f = 5
no_divisions = 10
T = np.array([np.linspace(t_i, t_f, (no_divisions) * (t_f - t_i) + 1)]).T

# sensor positions
s = np.array([[0, -1], 
              [1, 5]])

# Levenberg-Marquardt algorithm parameters
# initial guess
p = np.array([-1, 0])
v = np.array([0, 1])

# trust parameter lambda
lamb = 1

# tolerance parameter
epsilon = 1e-6

def get_displacement_from_sensor(p, v, t, sensor_index):
    """ Given the position and velocity of the target, get estimate of 
        distance to target number sensor_index

    Args:
        p: 2D NumPy array.
        v: 2D NumPy array.
        t: time instance (scalar)
        sensor_index: 1 or 2

    Returns:
        A scalar representing the displacement from the sensor
    """
    return (p + t * v) - s[sensor_index - 1]

def compute_gradient_g_t_i(p, v, t, sensor_index):
    """ Given the initial position and velocity of the target, get the gradient of the 
        function g_t_i such that f = sum(g_t_1 ** 2 + g_t_2 ** 2)
    Args:
        p: 2D NumPy array representing the current estimate of target's initial position
        v: 2D NumPy array representing the current estimate of target's velocity
        t: time instant (scalar)
        sensor_index: 1 or 2
    
    Returns:
        A 1D NumPy array representing the gradient
    """
    displacement = get_displacement_from_sensor(p, v, t, sensor_index)
    identities = np.concatenate((np.identity(2), t * np.identity(2)), axis = 0)

    return identities @ (displacement / np.linalg.norm(displacement))

def compute_g_t_i(p, v, t, measurement, sensor_index):
    """ Given the initial position and velocity of the target, get the value of the
        function g_t_i such that f = sum(g_t_1 ** 2 + g_t_2 ** 2)
    Args:
        p: 2D NumPy array representing the current estimate of target's initial position
        v: 2D NumPy array representing the current estimate of target's velocity
        t: time instant (scalar)
        sensor_index: 1 or 2

    Returns:
        A 1D NumPy array representing the gradient
    """
    displacement = get_displacement_from_sensor(p, v, t, sensor_index)
    return np.linalg.norm(displacement) - measurement
    

def compute_gradient_f_t(p, v, t, measurement_1, measurement_2):
    """ Given the initial position and velocity of the target, get the gradient of the 
        function representing the sum of the squared errors of the estimated distances
        to each sensor relative to the actual measured distances by each sensor (i.e., nabla(f_t(p_k, v_k)))
    Args:
        p: 2D NumPy array representing the current estimate of target's initial position
        v: 2D NumPy array representing the current estimate of target's velocity
        t: time instant (scalar)
        measurement_1: distance measured by sensor 1
        measurement_2: distance measured by sensor 2

    Returns:
        A 1D NumPy array representing the gradient
    """

    return 2 * compute_g_t_i(p, v, t, measurement_1, 1) * compute_gradient_g_t_i(p, v, t, 1) + \
           2 * compute_g_t_i(p, v, t, measurement_2, 2) * compute_gradient_g_t_i(p, v, t, 2)

def compute_f_t(p, v, t, measurement_1, measurement_2):
    """ Given the position and velocity of the target, get the sum of the 
        squared errors of the estimated distances to each sensor relative 
        to the actual measured distances by each sensor (i.e., f_t)
    Args:
        p: 2D NumPy array representing the current estimate of target's initial position
        v: 2D NumPy array representing the current estimate of target's velocity
        t: time instance (scalar)
        measurement_1: distance measured by sensor 1
        measurement_2: distance measured by sensor 2

    Returns:
        A scalar representing
    """
    return compute_g_t_i(p, v, t, measurement_1, 1) ** 2 + compute_g_t_i(p, v, t, measurement_2, 2) ** 2

def compute_gradient_f(p, v, r1, r2):
    """ Given the position and velocity of the target, get the gradient of the 
        function representing the sum of the squared errors of the estimated distances
        to each sensor relative to the actual measured distances by each sensor at all instants 
        (i.e., sum(nabla(f_t(p_k, v_k))))
     Args:
        p: 2D NumPy array representing the current estimate of target's initial position
        v: 2D NumPy array representing the current estimate of target's velocity
        r1: 1D NumPy array representing measurements of sensor 1
        r2: 1D NumPy array representing measurements of sensor 2

    Returns:
        A 1D NumPy array repreesenting grad(f)
    """
    # matrix where the entries of row i are (t_i, r1[t_i], r2[t_i])
    t_m_triples = np.hstack((T, r1.reshape(r1.size, 1), r2.reshape(r2.size, 1)))

    return np.apply_along_axis(lambda entry: compute_gradient_f_t(p, v, entry[0], entry[1], entry[2]),
                               arr = t_m_triples, axis = 1).sum(axis = 0)

def compute_f(p, v, r1, r2):
    """ Given the position and velocity of the target, get the sum of the 
        squared errors of the estimated distances to each sensor relative 
        to the actual measured distances by each sensor at all instants (i.e., sum(f_t))
    Args:
        p: 2D NumPy array representing the current estimate of target's initial position
        v: 2D NumPy array representing the current estimate of target's velocity
        r1: 1D NumPy array representing measurements of sensor 1
        r2: 1D NumPy array representing measurements of sensor 2

    Returns:
        A scalar representing f
    """
    # matrix where the entries of row i are (t_i, r1[t_i], r2[t_i])
    t_m_triples = np.hstack((T, r1.reshape(r1.size, 1), r2.reshape(r2.size, 1)))

    return np.apply_along_axis(lambda entry: compute_f_t(p, v, entry[0], entry[1], entry[2]),
                               arr = t_m_triples, axis = 1).sum(axis = 0)

def get_stacked_g_gradients(p, v):
    """ Get a matrix where the (2k + 1)-th and (2k + 2)-th row have nabla(g_k_1) transpose
        and nabla(g_k_2) transpose, respectively, for k = 0, ..., T
    Args:
        p: 2D NumPy array representing the current estimate of target's initial position
        v: 2D NumPy array representing the current estimate of target's velocity
    
    Returns:
        The matrix with the stacked transposed gradients
    """

    # matrix where the first and the second entries of row i have (t_i, 1) and (t_i, 2), respectively
    t_i_pairs = np.array([[t_i, i] for t_i in T.flatten() for i in range(1, 3)])
    #print(t_i_pairs)

    return np.apply_along_axis(lambda entry: compute_gradient_g_t_i(p, v, entry[0], int(entry[1])),
                                                arr = t_i_pairs, axis = 1)

def get_stacked_g_values(p, v, measurements_1, measurements_2):
    """ Get a column vector where the (2k + 1)-th and (2k + 2)-th entry have g_k_1
        and  g_k_2, respectively, for k = 0, ..., T
    Args:
        p: 2D NumPy array representing the current estimate of target's initial position
        v: 2D NumPy array representing the current estimate of target's velocity
    
    Returns:
        The matrix with the stacked transposed gradients
    """

    # matrix where the first and the second entries of row i have (t_i, 1) and (t_i, 2), respectively
    t_i_pairs = np.array([[t_i, i] for t_i in T.flatten() for i in range(1, 3)])

    # array where measurements from the first and second sensors are intertwined
    measurements = np.empty((measurements_1.size + measurements_2.size), dtype = measurements_1.dtype)
    measurements[0::2] = measurements_1
    measurements[1::2] = measurements_2

     # matrix where the first and the second entries of row i have (t_i, 1, measurement_1) 
     # and (t_i, 2, measurement_2), respectively
    t_i_m_triples = np.hstack((t_i_pairs, measurements.reshape(measurements.size, 1)))

    return np.apply_along_axis(lambda entry: compute_g_t_i(p, v, entry[0], entry[2], int(entry[1])),
                                                arr = t_i_m_triples, axis = 1)

# Read the data
measurements = loadmat('../inputs/measurements.mat')
r1 = measurements["r1"].flatten()
r2 = measurements["r2"].flatten()

function_values = [compute_f(p, v, r1, r2)]
gradient_norms = []

# Run Levenberg-Marquardt Algorithm for non-linear Least-Squares
while (True):

    # check stopping condition
    curr_grad = compute_gradient_f(p, v, r1, r2)
    gradient_norms.append(np.linalg.norm(curr_grad))

    if np.linalg.norm(curr_grad) < epsilon:
        break
    
    # Solve least squares problem
    grads_g = get_stacked_g_gradients(p, v)
    g_values = get_stacked_g_values(p, v, r1, r2)
    x = np.concatenate((p, v), axis = 0)

    while (True):

        # Construct A matrix and b vector of the corresponding least-squares problem
        diagonal_matrix = np.sqrt(lamb) * np.identity(4)
        A = np.concatenate((grads_g, diagonal_matrix), axis = 0)
        b = grads_g @ x - g_values
        b = np.concatenate((b, diagonal_matrix @ x), axis = 0)

        solution = np.linalg.lstsq(A, b, rcond = None)[0]
        tentative_p = solution[ : 2]
        tentative_v = solution[2 : ]

        function_values.append(compute_f(tentative_p, tentative_v, r1, r2))

        # check if the step was valid
        if function_values[-1] < function_values[-2]:
            p = tentative_p
            v = tentative_v
            lamb = lamb * 0.7
            break
        else:
            lamb = lamb * 2
            gradient_norms.append(np.linalg.norm(curr_grad))

plt.rcParams['text.usetex'] = True
plt.rc('font', family='serif')

# Plot cost function values across iterations
k_range = [i for i in range(1, len(function_values) + 1)]

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(k_range, function_values)

ax.set_xlabel('$k$')
ax.set_ylabel('cost function $f(x_k)$')

ax.set_xticks(k_range)
ax.grid()

plt.savefig("./output/cost_function_values.pdf")

# Plot gradient norm values across iterations
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(k_range, gradient_norms)

ax.set_xlabel('$k$')
ax.set_ylabel('$\|\\nabla f(x_k)\|_{2}$')

ax.set_xticks(k_range)

ax.set_yscale('log')
ax.grid()

plt.savefig("./output/gradient_norms.pdf")


