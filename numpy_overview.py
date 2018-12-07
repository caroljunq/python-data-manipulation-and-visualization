#  Examples from
# https://www.datacamp.com/community/tutorials/python-numpy-tutorial
# https://docs.scipy.org/doc/numpy-1.15.0/user/quickstart.html
import numpy as np
import matplotlib.pyplot as plt

# Arrays
array = np.array([1,2,3],dtype=np.int64)
# indicates the memory address of the first bytein the array
print(array.data)
print(array.dtype) # describes what the kind of elements
print(array.shape) # the dimension of the array

# Filling the array

# Create an array of ones
np.ones((3,4))

# Create an array of zeros
np.zeros((2,3,4),dtype=np.int16)

# Create an array with random values
np.random.random((2,2))

# Create an empty array
np.empty((3,2))

# Create a full array with 7s
np.full((2,2),7)

# Create an array of evenly-spaced values
np.arange(10,25,5)

# Create an array of evenly-spaced values
np.linspace(0,2,9)

# Create a identity matrix
np.eye(5)

# Create a identity matrix
np.identity(5)

# Load data from txt
#unpack = True for separate the arrays inside a matrix
x,y,z = np.loadtxt('./data/number.txt',
                    skiprows=1,
                    unpack=True)
# print(x)

# Missing value in txt
my_array = np.genfromtxt('./data/number_missing.txt',
                      skip_header=1,
                      filling_values=-999)
print(my_array)


# Save Numpy arrays
x = np.arange(0.0,5.0,1.0)
np.savetxt('./data/test.out', x, delimiter=',')

# Inspecting Numpy arrays


# Print the number of `my_array`'s dimensions
print(my_array.ndim)

# Print the number of `my_array`'s elements
print(my_array.size)

# Print information about `my_array`'s memory layout
print(my_array.flags)

# Print the length of one array element in bytes
print(my_array.itemsize)

# Print the total consumed bytes by `my_array`'s elements
print(my_array.nbytes)

print(len(my_array))

# Broadcasting it’s a mechanism that allows NumPy to work with arrays of
# different shapes when you’re performing arithmetic operations
# First off, to make sure that the broadcasting is successful, the dimensions of
# your arrays need to be compatible. Two dimensions are compatible when they are
# equal

## Arithmetic operations
# Sum
# Initialize `x`
x = np.ones((3,4))

# Check shape of `x`
print(x.shape)

# Initialize `y`
y = np.random.random((3,4))

# Check shape of `y`
print(y.shape)

# Add `x` and `y`
print(x + y)

np.add(x,y)
np.subtract(x,y)
np.multiply(x,y)
np.divide(x,y)
np.remainder(x,y)

ar=np.array([1,2,3])
ar=np.exp(ar)
print(ar)


# Aggregate functions
a2 = np.arange(1,101,1)
a2.sum()
a2.min()
a2.max(axis=0)
a2.cumsum(axis=0)
a2.mean()
# Median
np.median(a2)
# Correlation coefficient
np.corrcoef(a2)
# Standard deviation
np.std(a2)

#How To Subset, Slice, And Index Arrays
# Select the element at row 1 column 2
# print(my_2d_array[1][2])
#
# # Select the element at row 1 column 2
# print(my_2d_array[1,2])
#
# # Select the element at row 1, column 2 and
# print(my_3d_array[1,1,0])
#
#
# # Select items at index 0 and 1
# print(my_array[0:2])
#
# # Select items at row 0 and 1, column 1
# print(my_2d_array[0:2,1])
#
# # Select items at row 1
# # This is the same as saying `my_3d_array[1,:,:]
# print(my_3d_array[1,...])

# Boolean Slicing
a = np.random.random((3,2))
a[a > 0.5]

# Multiple Selection
# When it comes to fancy indexing, that what you basically do with it is the
# following: you pass a list or an array of integers to specify the order of the
# subset of rows you want to select out of the original array
# my_2d_array = np.array([[5, 6, 7, 8],
#    [1, 2, 3, 4],
#    [5, 6, 7, 8],
#    [1, 2, 3, 4]])
print(my_array[[1, 0, 1, 0],[0, 1, 2, 0]])

# Getting information
np.info(a)

# Transpose Arrays
# Print `my_2d_array`
a = np.random.random((3,2))

# Transpose `my_2d_array`
print(np.transpose(a))

# Or use `T` to transpose `a`
print(a.T)

# Resizing --> you change the data in the array
# Print the shape of `x`
print(x.shape)

# Resize `x` to ((6,4))
# Fill with copies of elements
np.resize(a, (6,4))

# Fill with 0ss
# Try out this as well
a.resize((6,4))

# Print out `a`
print(a)

# Reshaping
# Print the size of `x` to see what's possible
print(a.size)

# Reshape `a` to (2,6)
# print(a.reshape((2,6)))

# Flatten `a`
z = a.ravel()

# Print `z`
print(z)

# Appending Arrays
# Append a 1D array to your `my_array`
# new_array = np.append(my_array, [7, 8, 9, 10])
#
# # Print `new_array`
# print(new_array)
#
# # Append an extra column to your `my_2d_array`
# new_2d_array = np.append(my_2d_array, [[7], [8]], axis=1)
#
# # Print `new_2d_array`
# print(new_2d_array)

# Inserting and deleting
# Insert `5` at index 1
np.insert(my_array, 1, 5)

# Delete the value at index 1
np.delete(my_array,[1,1])

##  Visualizing data
np.histogram(my_array)

# Initialize your array
# my_3d_array = np.array([[[1,2,3,4], [5,6,7,8]], [[1,2,3,4], [9,10,11,12]]], dtype=np.int64)
#
# # Pass the array to `np.histogram()`
# print(np.histogram(my_3d_array))
#
# # Specify the number of bins
# print(np.histogram(my_3d_array, bins=range(0,13)))

# Graph Representation
plt.hist(my_array.ravel(), bins=range(0,13))

# Add a title to the plot
plt.title('Frequency of My 3D Array Elements')

# Show the plot
plt.show()
