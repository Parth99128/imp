


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

age = np.arange(1, 21).reshape(-1, 1)
price = np.array([98, 95, 88, 83, 81, 79, 75, 71, 69, 65,
                  62, 58, 52, 47, 43, 34, 28, 23, 21, 20])


model = LinearRegression()
model.fit(age,price)
predicted_price = model.predict(age)

print("slope:",model.coef_[0])
print("intercept:",model.intercept_)
print("predicted price:",predicted_price)

plt.scatter(age,price, label = "Actual price")
plt.plot(age,predicted_price,label = "Regression line")
plt.xlabel("age")
plt.ylabel("price")
plt.title("linear regression")
plt.legend()
plt.show()






