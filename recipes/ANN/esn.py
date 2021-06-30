# -----------------------------------------------------------------------------
# Copyright 2019 (C) Nicolas P. Rougier & Anthony Strock
# Released under a BSD two-clauses license
#
# References: Jaeger, Herbert (2001) The "echo state" approach to analysing and
#             training recurrent neural networks. GMD Report 148, GMD - German
#             National Research Institute for Computer Science
# -----------------------------------------------------------------------------
import numpy as np

# Parameters
shape = 1,1000,1
bias = 1.0
leak = 0.5
warmup = 100
radius = 1.25
epsilon = 1e-8
rng = np.random
rng.seed(1)

# Build reservoir
W_in  = 0.5*rng.uniform(-1, +1, (shape[1], 1+shape[0]))
W_rc  = 0.5*rng.uniform(-1, +1, (shape[1], shape[1]))
W_rc *= radius / np.max(np.abs(np.linalg.eigvals(W_rc)))

# Get data
data = np.load('mackey-glass.npy')
train_data = data[:2000,np.newaxis].ravel()
test_data = data[2000:4000,np.newaxis]

# Run & collect extended states (input + reservoir)
X = np.zeros((len(train_data), shape[1]))
x = X[0]
for i in range(1,len(train_data)):
    u = bias, train_data[i-1]
    z = np.dot(W_in, u) + np.dot(W_rc, x)
    x = (1-leak)*x + leak*np.tanh(z)
    X[i] = x

# Learn (Ridge regression)
X, Y = X[warmup:], train_data[warmup:]
W_out = np.dot(np.dot(Y.T,X),
               np.linalg.inv(np.dot(X.T,X) + epsilon*np.eye(shape[1]))) 

# Test (generative mode)
outputs = []
x, y = X[-1], Y[-1]
for i in range(len(test_data)):
    u = bias, y
    z = np.dot(W_in, u) + np.dot(W_rc, x)
    x = (1-leak)*x + leak*np.tanh(z)
    y = np.dot(W_out, x)
    outputs.append(y)
    
# Display results
import matplotlib.pyplot as plt
plt.figure(figsize=(12,3))
plt.plot(data[2000:4000], label="Actual signal")
plt.plot(outputs, label="Generated signal")
plt.legend()
plt.show()
