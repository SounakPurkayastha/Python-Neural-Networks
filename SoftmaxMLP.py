import numpy as np 

X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
y = np.array([[1,0],[0,1],[0,1],[1,0]])

np.random.seed(1)

syn0 = np.random.random((X.shape[1],4))
syn1 = np.random.random((4,3))
syn2 = np.random.random((3,2))

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x),axis=1).reshape(-1,1)

def sigmoid(x):
	return 1/(1+np.exp(-x))

eta = 0.01
n_epochs = 30000

for e in range(n_epochs):
	l0 = X
	l1 = sigmoid(np.dot(l0,syn0))
	l2 = sigmoid(np.dot(l1,syn1))
	l3 = softmax(np.dot(l2,syn2))
	l3_error = y - l3
	l3_delta = l3_error
	l2_error = np.dot(l3_delta,syn2.T)
	l2_delta = l2_error * l2 * (1-l2)
	l1_error = np.dot(l2_delta,syn1.T)
	l1_delta = l1_error * l1 * (1-l1)
	syn0 += eta * np.dot(l0.T,l1_delta)
	syn1 += eta * np.dot(l1.T,l2_delta)
	syn2 += eta * np.dot(l2.T,l3_delta)
	if e%1000==0:
		print('Error:',np.mean(np.abs(l3_error)))
print(l3)