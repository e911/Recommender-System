from numpy import *

def costFunction(serialized_vector,Y,R,num_users,num_movies,num_features,l):
	
	X= reshape(serialized_vector[:num_movies*num_features],(num_movies,num_features),order='F')
	Theta= reshape(serialized_vector[num_movies*num_features:],(num_users,num_features),order='F')


	J=0;
	X_gradient =zeros(shape(X))
	Theta_gradient = zeros(shape(Theta))

	error =((dot(X, Theta.T)-Y)*R)
	squaredError = error**2
	x_sq = X**2
	theta_sq = Theta **2
	J=((1/2)*squaredError.sum(axis=None))+((l/2)*theta_sq.sum(axis=None))+((l/2)*x_sq.sum(axis=None))
	X_gradient = dot(error,Theta) + (l*X) 
	Theta_gradient = dot(error.T,X) + (l*Theta) 
	grad = hstack((X_gradient.ravel('F'), Theta_gradient.ravel('F')))
	return J,grad
