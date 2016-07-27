import pprint

def load_new_data():

	movieList = []

	## Read the fixed movieulary list
	with open('movie_ids.txt') as fid:
		for line in fid:
			movieName = line.split(' ', 1)[1].strip()
			movieList.append(movieName)

	return movieList
#checking how it reads the manes from files :v
# fidata= open('movie.txt','w')
# fidata.write(pprint.pformat(movieList))