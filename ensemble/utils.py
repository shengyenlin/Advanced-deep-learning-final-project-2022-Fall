import pickle

def load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def standardization(data, max=None, min=None):
    if (max == None) or (min == None):
        max = max(list(data.values()))
        min = min(list(data.values()))

    standardized_data = {key: ((value - min) / (max - min)) for key, value in data.items()}
    return standardized_data