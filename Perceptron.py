import numpy as np 

X = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
Y = np.array([[0],[1],[1],[1]])

w = np.random.random((X.shape[1],1))
Y_hat = np.heaviside(np.matmul(X,w),0)
error = np.sum(Y!=Y_hat)

while error!=0:
	for x,y in zip(X,Y):
		y_hat = np.heaviside(np.matmul(x,w),0)
		if y==1 and y_hat==0:
			w = w + x.reshape(-1,1)
		if y==0 and y_hat==1:
			w = w - x.reshape(-1,1)
	error = np.sum(Y!=np.heaviside(np.matmul(X,w),0))

print(np.heaviside(np.matmul(X,w),0))