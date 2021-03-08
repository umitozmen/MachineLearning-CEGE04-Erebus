import pandas as pd


def readDatabaseFile(filePath):
    # read csv data
    with open(filePath) as f:
        # load reads the csv db as a dictionary with
        # the data as a list of lists at key "data"
        dataFrame = pd.read_csv(f)
        f.close()

    return dataFrame


if __name__ == "__main__":
    readDatabaseFile("online_shoppers_intention.csv")
