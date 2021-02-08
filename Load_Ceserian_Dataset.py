import arff
import numpy as np
import pandas as pd


def readCeserianFile(filePath):
    # read arff data
    with open(filePath) as f:
        # load reads the arff db as a dictionary with
        # the data as a list of lists at key "data"
        dataDictionary = arff.load(f)
        f.close()

    # extract data and convert to numpy array
    arffData = np.array(dataDictionary['data'])
    arffAttributes = [i[0] for i in dataDictionary['attributes']]

    return pd.DataFrame(arffData, columns=arffAttributes) 


if __name__ == "__main__":

    readCeserianFile("caesarian.csv.arff")
