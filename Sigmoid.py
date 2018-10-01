import numpy as np

X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
Y = np.array([[0],[1],[1],[1]])

w = np.random.random((X.shape[1],1))

n_epochs = 7000
learning_rate = 0.01

def sigmoid(x):
	return 1/(1+np.exp(-x))

for e in range(n_epochs):
	for x,y in zip(X,Y):
		y_hat = sigmoid(np.dot(x,w))
		error = - y*np.log(y_hat) - (1-y)*np.log(1-y_hat)
		grad_C = x.reshape(-1,1) * (y-y_hat)
		del_w = learning_rate * grad_C 
		w += del_w
		if e%1000==0:
			print('Error:',np.mean(error))

print(sigmoid(np.matmul(X,w)))