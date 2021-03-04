# Attribute Information:
#
# We choose age, delivery number, delivery time, blood pressure and heart status.
# We classify delivery time to Premature, Timely and Latecomer. As like the delivery time we consider blood pressure in
# three statuses of Low, Normal and High moods. Heart Problem is classified as apt and inept.
#
# @attribute 'Age' { 22,26,28,27,32,36,33,23,20,29,25,37,24,18,30,40,31,19,21,35,17,38 }
# @attribute 'Delivery number' { 1,2,3,4 }
# @attribute 'Delivery time' { 0,1,2 } -> {0 = timely , 1 = premature , 2 = latecomer}
# @attribute 'Blood of Pressure' { 2,1,0 } -> {0 = low , 1 = normal , 2 = high }
# @attribute 'Heart Problem' { 1,0 } -> {0 = apt, 1 = inept }
#
# @attribute Caesarian { 0,1 } -> {0 = No, 1 = Yes }
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

    print(readCeserianFile("caesarian.csv.arff"))
