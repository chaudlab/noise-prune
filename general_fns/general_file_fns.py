'''February 18th 2020
Functions to load and save files. '''

import pickle
import glob


def load_file_from_pattern(file_pattern):
    file_matches = glob.glob(file_pattern)
    if len(file_matches) > 1:
        print('Multiple matches. Using the first one')
    if len(file_matches) == 0:
        print('No file found')
        return
    fname = file_matches[0]
    data = load_pickle_file(fname)
    return data, fname


def load_pickle_file(filename):
    fr = open(filename, 'rb')
    data = pickle.load(fr)
    fr.close()
    return data


def save_pickle_file(data, filename):
    fw = open(filename, 'wb')
    pickle.dump(data, fw)
    fw.close()
    return 1
