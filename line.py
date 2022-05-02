from cProfile import label
import matplotlib.pyplot as plt
import numpy as np

#input values
x = np.array([10,9,2,15,10,16,11,16])
y = np.array([95,80,10,50,45,98,38,93])

for i in range(0,len(x)):
	plt.plot(x[i],y[i],"gX")

#slope intercept cal
slope, intercept = np.polyfit(x, y, 1)
y= slope*x + intercept

#plotting co-ordinates

for i in range(0,len(x)):
	plt.plot(x[i],y[i],"bo")

plt.plot(x, y, '-r', label='y=mx+b')
plt.ylabel('Risk Score on a scale of 0-100')
plt.xlabel('NO. of hours spent on Driving ')
plt.title("Linear Regression")
plt.show()

