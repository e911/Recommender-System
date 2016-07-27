from numpy import *
from scipy.io import loadmat
from scipy.optimize import minimize
#from matplotlib.pyplot import *

from costFunction import costFunction
from checkCostFunction import checkCostFunction
from load_new_data import load_new_data
from normalization import normalization


def serialize(*args):
    return hstack(a.ravel('F') for a in args)


#load movie data
movie_data=loadmat("ex8_movies.mat")
Y= movie_data['Y'] #get_rating 
R= movie_data['R'] #if_rating 

#plot the data
# fig=figure()
# imshow(Y,aspect=.4)
# ylabel("Movies")
# xlabel("User")
# fig.show()


#previous data weights X amd theta
pre_movie_data=loadmat("ex8_movieParams.mat")
X=pre_movie_data['X']
Theta=pre_movie_data['Theta']

# for all data i think
# num_users=pre_movie_data['num_users'][0][0]
# print (num_users)
# num_movies=pre_movie_data['num_movies'][0][0]
# num_features= pre_movie_data['num_features'][0][0]
# print(num_movies)
# print(num_features)

num_users, num_movies, num_features = 4, 5, 3
X = X[:num_movies,:num_features]
Theta = Theta[:num_users,:num_features]
Y = Y[:num_movies,:num_users]
R = R[:num_movies,:num_users]


J,_ = costFunction(serialize(X,Theta),Y,R,num_users,num_movies,num_features,1.5)


print ('Cost at loaded parameters: %f ' % J)



checkCostFunction(1.5)

#give input ratings
#these thinga are done from the user side 

movieList = load_new_data()
my_ratings = zeros(len(movieList))
#give ratings to movie
my_ratings[1-1]=4
my_ratings[234-1]=3

my_ratings[7-1] = 3
my_ratings[12-1]= 5
my_ratings[54-1] = 4
my_ratings[64-1]= 5
my_ratings[66-1]= 3
my_ratings[69-1] = 5
my_ratings[183-1] = 4

my_ratings[226-1] = 5
my_ratings[355-1]= 5


print ("New ratings")
for rating, name in zip(my_ratings,movieList):
	if(rating>0):
		print ("Rated %d for movie %s" %(rating,name))


print("New Data training set Loading.... \n\n\n")

ex8_movies = loadmat('ex8_movies.mat')
Y = ex8_movies['Y']
R = ex8_movies['R']

#add the data

Y= column_stack((my_ratings,Y))
R= column_stack((my_ratings!=0,R))

Ynorm,Ymean= normalization(Y,R)
num_users = size(Y, 1)
num_movies = size(Y, 0)
num_features = 10



X = random.randn(num_movies, num_features)
Theta = random.randn(num_users, num_features)


initial_parameters = serialize(X, Theta)

l = 10
extra_args = (Y, R, num_users, num_movies, num_features,l)
import sys
def callback(p): sys.stdout.write('.')

res = minimize(costFunction, initial_parameters, extra_args, method='CG',
               jac=True, options={'maxiter':100}, callback=callback)
theta = res.x
cost = res.fun

print ("\nFinal cost: %e" % cost)

X = reshape(theta[:num_movies*num_features], (num_movies, num_features), order='F')
Theta = reshape(theta[num_movies*num_features:], (num_users, num_features), order = 'F')

print ('Recommender system learning completed.')



p = dot(X, Theta.T)

my_predictions = p[:,0]

movieList = load_new_data()

ix = argsort(my_predictions)
print ('\nTop recommendations for you:')
for j in ix[:-11:-1]:
    print ('Predicting rating %.1f for movie %s' % (my_predictions[j], movieList[j]))

print ('\n\nOriginal ratings provided:')
for rating, name in zip(my_ratings, movieList):
    if rating > 0:
        print ('Rated %d for %s' % (rating, name))

