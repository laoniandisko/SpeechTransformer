import pickle
with open("aishell.pickle", 'rb') as file:
    load = pickle.load(file)
print(load)