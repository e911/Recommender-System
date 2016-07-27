from numpy import *

def normalization(Y,R):
	m,n=shape(Y)
	Y_mean = zeros(m)
	Y_norm = zeros(shape(Y))
	for i in range(m):
		idx = where(R[i,:]==1)
		Y_mean[i] = mean(Y[i,idx])
		Y_norm[i,idx] = Y[i,idx]-Y_mean[i]

	return Y_norm,Y_mean