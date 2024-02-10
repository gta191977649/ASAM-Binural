import theano
import theano.tensor as T
import numpy as np

# Check if Theano is using GPU
print("Using device:", theano.config.device)

# Define a Theano operation
x = T.matrix('x')
y = T.matrix('y')
z = T.dot(x, y)

# Compile the function
f = theano.function([x, y], z)

# Test data
test_x = np.random.rand(1000, 1000).astype(theano.config.floatX)
test_y = np.random.rand(1000, 1000).astype(theano.config.floatX)

# Perform the operation
result = f(test_x, test_y)

print("Matrix multiplication performed, result shape:", result.shape)
