import matplotlib.pyplot as plt

# Data
s = [0.31, 0.33, 0.47, 0.56, 0.58, 0.62, 0.69, 0.7, 0.79, 0.85]
n = len(s)

# Minimum and maximum values
y_min = min(s)
y_max = max(s)
print(y_min)
print(y_max)
# Epsilon and xi parameters
epsilon = 2.2e-16
xi = 0.02

# Calculate m, q, and z
m = (y_max - y_min + 2 * xi) / (n - 1)
q = y_min - m - xi
z = [m * (i+1) + q for i in range(n)]  # List comprehension for efficiency
print(z)
# Calculate jb and jt (integer casting)
jb = int((0.52 - q) / m) - 1
jt = int((0.76 - q) / m)

# x-axis for plotting (indices)
x = range(1, n+1)  # Adjust for 1-based indexing if needed

# Create the plot
plt.figure()  # Create a new figure window

plt.plot(x, z, label='z(k)')  # Plot z(k)
plt.plot(x, s, marker='*',markersize=7, linestyle='None', label='s(k)')  # Plot s(k) with markers
plt.plot(x, [0.52 * i / i for i in x], label='0.52')  # Plot 0.52 line with slope 1
plt.plot(x, [0.76 * i / i for i in x], label='0.76')  # Plot 0.76 line with slope 1

plt.xlabel("k-vector elements")
plt.ylabel("Sorted database s=y(I)")
plt.title("Example of k-vector construction")
plt.grid(True)
plt.legend()
plt.savefig("kvector.png")
plt.show()
