import pickle

def create_pickle(name, train_obj):
    pickle_out = open(name, "wb")
    pickle.dump(train_obj, pickle_out)
    pickle_out.close()


def load_pickle(name):
    pickle_in = open(name, "rb")
    obj = pickle.load(pickle_in)
    return obj



