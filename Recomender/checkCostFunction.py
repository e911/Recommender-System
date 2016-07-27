from numpy import *
from costFunction import costFunction

def checkCostFunction(l):
	# X_t = random.rand(943,10)
	# Theta_t = random.rand(1682,10)

	X_t = random.rand(4,3)
	Theta_t = random.rand(5,3)


	Y=dot(X_t,Theta_t.T)
	Y[random.rand(*shape(Y))>0.5]=0
	R=where(Y==0,0,1)

	X = random.randn(*shape(X_t))
	Theta= random.randn(*shape(Theta_t))
	num_users =size(Y,1)
	num_movies = size(Y,0)
	num_features = size(Theta_t,1)
	

	num_gradient = computeNumericaGradient(lambda t: costFunction(t,Y,R,num_users,num_movies,num_features,l),hstack((X.ravel('F'),Theta.ravel('F'))))
	cost,gradient = costFunction(hstack((X.ravel('F'),Theta.ravel('F'))),Y,R,num_users,num_movies,num_features,l)
	
	print (column_stack((num_gradient,gradient)))
    
	diff = linalg.norm(num_gradient-gradient) / linalg.norm(num_gradient+gradient)
	#small difference
	print ('\nRelative Difference: %g' % diff)


def computeNumericaGradient(J,theta):
	e=1e-4
	num_gradient =zeros(shape(theta))
	plusminus_value=zeros(shape(theta))

	for p in ndindex(shape(theta)):
		plusminus_value[p]=e
		loss1,_ =J(theta-plusminus_value)
		loss2,_ =J(theta+plusminus_value)

		#calculate the differentiant : plus del - minus del/ 2* smallvalue
		num_gradient[p]= (loss2-loss1)/(2*e)
		plusminus_value[p]=0
	return num_gradient	

