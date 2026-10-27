# Practical 3: Gradient Descent for y = (x + 3)^2

import matplotlib.pyplot as plt

x = 2.0
learning_rate = 0.1
iterations = 30
loss_values = []

for i in range(iterations):
    y = (x + 3) ** 2
    gradient = 2 * (x + 3)  # derivative of (x + 3)^2
    loss_values.append(y)
    x = x - learning_rate * gradient

print("Minimum x (approximately):", x)
print("Minimum y (approximately):", (x + 3) ** 2)

plt.plot(range(iterations), loss_values, marker="o")
plt.xlabel("Iteration")
plt.ylabel("y")
plt.title("Gradient Descent")
plt.show()

