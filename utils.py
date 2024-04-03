import json


def save_json(data, path):
    with open(path, "w") as js:
        json.dump(data, js)


def load_json(path):
    with open(path, "r") as js:
        data = json.load(js)
    return data
