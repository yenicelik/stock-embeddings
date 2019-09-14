"""
    Extracts the embeddings from the keras model.
"""
import numpy as np
import pandas as pd

from dl.data_loader import dataloader
from dl.model.nn.baseline import BaselineModel

if __name__ == "__main__":

    is_dev = True

    df, encoder_date, encoder_label, decoder_date, decoder_label = dataloader.import_data(development=is_dev)
    market_df = dataloader.preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    numerical_feature_cols = list(market_df.columns[response_col + 1:])
    model = BaselineModel(
        encoder_label,
        number_of_numerical_inputs=len(numerical_feature_cols)
    )

    if is_dev:
        checkpoint = "/Users/david/deeplearning/data/model/BaselineModel_keras_dev.hdf5"
    else:
        checkpoint = "/Users/david/deeplearning/data/model/BaselineModel_keras_e1_d50.hdf5"
    model.keras_model.load_weights(checkpoint)
    model.keras_model.summary()

    embedding_layer = model.keras_model.layers[3].get_weights()[0]
    print("Embedding layer is: ", embedding_layer.shape)

    # Also save the labels somewhere

    from urllib.request import urlopen
    from lxml.html import parse
    import time

    '''
    Returns a tuple (Sector, Indistry)
    Usage: GFinSectorIndustry('IBM')
    '''


    def GFinSectorIndustry(name):
        tree = parse(urlopen('http://www.google.com/finance?&q=' + name))
        time.sleep(2)
        return tree.xpath("//a[@id='sector']")[0].text, tree.xpath("//a[@id='sector']")[0].getnext().text

    print("Label encoder is: ", encoder_label)

    labels = []
    for x in encoder_label.keys():
        sector, _ = GFinSectorIndustry(x)
        print("Sector is: ", sector)
        labels.append((x, sector))

    label_df = pd.DataFrame(labels, index=None, columns=None)
    label_df.to_csv("./embedding_labels.tsv", sep="\t", index=False, header=False)
    np.savetxt("./embeddings.tsv", embedding_layer, delimiter="\t")

    # print("Keras model: ", keras_model)
