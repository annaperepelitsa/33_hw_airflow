# <YOUR_IMPORTS>
import pandas as pd
import json
import os
import dill
import csv
import fnmatch


def predict():
    # read model from pickle file and unpickle it
    path = os.environ.get('PROJECT_PATH', '.')

    list_files = fnmatch.filter(os.listdir(f'{path}/data/models'), '*.pkl')
    newest_file = max(list_files, key=lambda x: os.path.getmtime(f'{path}/data/models/{x}'))

    with open(f'{path}/data/models/{newest_file}', 'rb') as file:
        model_from_pickle = dill.load(file)

    # find path to json files from test folder
    # and prediction
    pred_cat = []
    for filename in os.listdir(f'{path}/data/test'):
        with open(f'{path}/data/test/{filename}') as json_file:
            data = json.load(json_file)
            df = pd.DataFrame.from_dict([data])
            y = model_from_pickle.predict(df)
            pred_cat.append(y)

    # writing results into csv-file
    with open(f'{path}/data/predictions/prediction_test.csv', 'w') as file:
        writer = csv.writer(file)
        for elem in pred_cat:
            writer.writerow(elem)


if __name__ == '__main__':
    predict()
