"""
    Extracts the embeddings from the keras model.
"""
import numpy as np
import pandas as pd

from dl.data_loader import dataloader
from dl.model.nn.baseline import BaselineModel
from dl.training.params import params

import requests


def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

    time.sleep(2)
    result = requests.get(url).json()

    for x in result['ResultSet']['Result']:
        if x['symbol'] == symbol:
            return x['name']


if __name__ == "__main__":

    params.development = False

    df, encoder_date, encoder_label, decoder_date, decoder_label = dataloader.import_data()
    market_df = dataloader.preprocess(df)

    response_col = market_df.columns.get_loc("ReturnOpenNext1")
    numerical_feature_cols = list(market_df.columns[response_col + 1:])
    model = BaselineModel(
        encoder_label,
        number_of_numerical_inputs=len(numerical_feature_cols)
    )

    if params.development:
        print("Development!")
        checkpoint = "/Users/david/deeplearning/data/model/BaselineModel_keras_dev.hdf5"
    else:
        checkpoint = "/Users/david/deeplearning/data/model/BaselineModel_keras.hdf5"
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

    symbol2df = pd.read_csv("/Users/david/deeplearning/data/secwiki_tickers.csv")
    print("Symbols df are: ", symbol2df.columns)
    symbol2df = symbol2df[['Ticker', 'Name', 'Sector', 'Industry']]

    def GFinSectorIndustry(name):
        # tree = parse(urlopen('http://www.google.com/finance?&q=' + name))
        # time.sleep(2)
        # return tree.xpath("//a[@id='sector']")[0].text, tree.xpath("//a[@id='sector']")[0].getnext().text
        company = get_symbol("MSFT")
        return company

    print("Label encoder is: ", encoder_label)

    # Translate the name and ticket symbol

    labels = []
    for x in encoder_label.keys():
        # info = GFinSectorIndustry(x)
        # print("Info is: ", info)
        # labels.append((x, info))
        labels.append(x)

    label_df = pd.DataFrame(labels, index=None, columns=None)
    label_df.to_csv("/Users/david/deeplearning/data/model/embedding_labels.tsv", sep="\t", index=False, header=False)
    np.savetxt("/Users/david/deeplearning/data/model/embeddings.tsv", embedding_layer, delimiter="\t")

    # print("Keras model: ", keras_model)

