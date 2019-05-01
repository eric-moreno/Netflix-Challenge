import pandas as pd
import numpy as np
import gc

def make_sets(data, indices):
    base = data[indices.iloc[data.index, 0] == 1]
    valid = data[indices.iloc[data.index, 0] == 2]
    #hidden = data[indices.iloc[data.index, 0] == 3]
    #probe = data[indices.iloc[data.index, 0] == 4]
    qual = data[indices.iloc[data.index, 0] == 5]

    base.reset_index(drop=True, inplace=True)
    #valid.reset_index(drop=True, inplace=True)
    #hidden.reset_index(drop=True, inplace=True)
    #probe.reset_index(drop=True, inplace=True)
    qual.reset_index(drop=True, inplace=True)

    #return base, valid, hidden, probe, qual
    #return base, qual
    return base, qual

def generate_movie_table(data):
    movies = []

    movie = 1
    currRatings = []

    for index, row in data.iterrows():
        if (row['Movie'] == movie):
            currRatings.append(row['Rating'])
        else:
            mean = np.mean(currRatings)
            #std = np.std(currRatings)
            count = len(currRatings)
            movies.append([movie, mean, 0, count])

            movie += 1
            currRatings = []

            print([movie, mean, 0, count])

    df = pd.DataFrame(movies)
    df.to_csv('E:\\Documents\\Caltech\\CS156\\mu\\movies.dta', header=None, index=None, sep=' ')

def generate_user_table(data):
    users = []

    user = 1
    currRatings = []

    for index, row in data.iterrows():
        if (row['User'] == user):
            currRatings.append(row['Rating'])
        else:
            mean = np.mean(currRatings)
            #std = np.std(currRatings)
            count = len(currRatings)
            users.append([user, mean, 0, count])

            user += 1
            currRatings = []

            if(user % 10000 == 0):
                print([user, mean, 0, count])

    df = pd.DataFrame(users)
    df.to_csv('E:\\Documents\\Caltech\\CS156\\users.dta', header=None, index=None, sep=' ')


def generate_new_user_table(data):
    users = []

    user = 1
    currRatings = []

    for index, row in data.iterrows():
        if (row['User'] == user):
            currRatings.append(row['Rating'])
        else:
            #mean = np.mean(currRatings)
            count = len(currRatings)
            u, c = np.unique(currRatings, return_counts=True)
            counts = [0, 0, 0, 0, 0]

            for i in range(len(u)):
                counts[u[i] - 1] += c[i]

            for i in range(len(counts)):
                counts[i] = counts[i] / count

            users.append([user, counts[0], counts[1], counts[2], counts[3], counts[4], count])

            user += 1
            currRatings = []

            if(user % 10000 == 0):
                print([user, counts[0], counts[1], counts[2], counts[3], counts[4], count])

    df = pd.DataFrame(users)
    df.to_csv('E:\\Documents\\Caltech\\CS156\\usersLinear.dta', header=None, index=None, sep=' ')

def predict(qual):
    predictions = []

    for index, row in qual.iterrows():
        movie = row['Movie']
        user = row['User']
        date = row['Date']

        prediction = 3.0
        predictions.append(prediction)

    df = pd.DataFrame(predictions)
    df.to_csv('E:\\Documents\\Caltech\\CS156\\predictions.dta', header=None, index=None, sep=' ')


dataset = pd.read_csv('E:\\Documents\\Caltech\\CS156\\um\\all.dta', delim_whitespace=True, header=None, names =['User', 'Movie', 'Date', 'Rating'])
indices = pd.read_csv('E:\\Documents\\Caltech\\CS156\\um\\all.idx', delim_whitespace=True, header=None)

base, qual = make_sets(dataset, indices)

generate_new_user_table(base)
#del dataset
#del indices
#gc.collect()

#umdta = pd.read_csv('E:\\Documents\\Caltech\\CS156\\mu\\all.dta', delim_whitespace=True, header=None, names =['User', 'Movie', 'Date', 'Rating'])
#umidx = pd.read_csv('E:\\Documents\\Caltech\\CS156\\mu\\all.idx', delim_whitespace=True, header=None)

#baseum, qualum = make_sets(umdta, umidx)

#del umdta
#del umidx
#gc.collect()





#generate_user_table(base)

#generate_movie_table(base)


#f= open("averages.dta","w+")
#for i in base:
#    string = str('%.3f'%(i)) + '\n'
#    f.write(string)
#f.close


#print(find_average_movie_rating(base, 10))

#for index, row in qual.iterrows():
#    movie = row['Movie']
#    user = row['User']


#full_array = np.zeroes((480189, 17770))




