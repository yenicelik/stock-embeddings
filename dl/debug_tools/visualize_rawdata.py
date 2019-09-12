"""
    Some debug tools, such as visualizations etc.
"""

from dl.data_loader import import_data
from dl.data_loader import preprocess

import matplotlib.pyplot as plt
plt.style.use('classic')

import seaborn as sns
sns.set()


def abb_plot():
    """
        Plots the following items:
        - response variable
        - OHLC
        -
    :return:
    """
    print("Starting plot")
    market_df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=True)
    market_df = preprocess(market_df)
    market_df.head(10)

    abb_label = encoder_label.get("abb")
    abb_df = market_df[market_df.Label == abb_label].reset_index(drop=True)

    abb_df = abb_df.head(500)
    print(abb_df)

    # g = sns.factorplot(x="Date", y="", hue='cols', data=df)
    # print(g)

    plt.plot(abb_df.Date, abb_df.Open)

    plt.xlabel('Timestep')
    plt.ylabel('Stock value in $ (Open)')
    plt.title('ABB Stock value in $ per timestep (Sample)')

    plt.show()
    # plt.legend('ABCDEF', ncol=2, loc='upper left')

def abb_plot_response_variable():
    """
        Plots the following items:
        - response variable
        - OHLC
        -
    :return:
    """
    print("Starting plot")
    market_df, encoder_date, encoder_label, decoder_date, decoder_label = import_data(development=True)
    market_df = preprocess(market_df)
    market_df.head(10)

    abb_label = encoder_label.get("abb")
    abb_df = market_df[market_df.Label == abb_label].reset_index(drop=True)

    abb_df = abb_df.head(500)
    print(abb_df)

    # g = sns.factorplot(x="Date", y="", hue='cols', data=df)
    # print(g)

    plt.plot(abb_df.Date, abb_df.ReturnOpenPrevious5)

    plt.xlabel('Timestep')
    plt.ylabel('5-Day-Change in $')
    plt.title('5-Day-Change in ABB Stock value in $ per timestep (Sample)')

    plt.show()
    plt.clf()

    plt.plot(abb_df.Date, abb_df.ReturnOpenPrevious1)

    plt.xlabel('Timestep')
    plt.ylabel('1-Day-Change in $')
    plt.title('1-Day-Change in ABB Stock value in $ per timestep (Sample)')

    plt.show()
    plt.clf()

    plt.plot(abb_df.Date, abb_df.ReturnOpenNext1)

    plt.xlabel('Timestep')
    plt.ylabel('Response Variable 1-Day-Change in $')
    plt.title('Response Variable 1-Day-Change in ABB Stock value in $ per timestep (Sample)')

    plt.show()
    plt.clf()
    # plt.legend('ABCDEF', ncol=2, loc='upper left')



if __name__ == "__main__":
    print("Plotting...")

    # abb_plot()
    abb_plot_response_variable()