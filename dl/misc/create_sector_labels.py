"""
    Creates sector labels from the tsv files
"""
import pandas as pd

if __name__ == "__main__":
    print("Apple is: ")
    df = pd.read_csv("/Users/david/deeplearning/dl/misc/embedding_sector_labels.csv", header=None, index_col=None)
    print(df.head(10))

    df.iloc[:,1] = df.iloc[:,1] // 10000

    print("Ducpliated:", df[df.duplicated()])
    print(len(df.iloc[:,1].unique()))
    print(len(df))

    # df = df.iloc[:, :2]

    df.to_csv("/Users/david/deeplearning/dl/misc/embedding_sector_labels___.tsv", sep="\t", index=False, header=False)