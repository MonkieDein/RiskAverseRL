import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def getdf(filename):
    # Read in the data text file
    with open(filename, "r") as file:
        data = file.readlines()
    data = [line for line in data if line != "\n"]
    # Extract the column names from the first row
    column_names = data[0].replace(" ", "").strip().split("&")

    # Extract the data rows from the remaining lines
    data_rows = [row.replace(" ", "").strip().split("&") for row in data[1:]]
    data_rows = [[row[0]]+[float(v) for v in row[1:]] for row in data_rows]
    # Create a pandas DataFrame from the column names and data rows
    df = pd.DataFrame(data_rows, columns=column_names)
    # .set_index(column_names[0])
    return df


def overlapped_bar(df, show=False, width=0.9, alpha=.5,
                   title='', xlabel='', ylabel='', domain="", offset=0.0):
    """Like a stacked bar chart except bars on top of each other with transparency"""
    xlabel = xlabel or df.index.name
    N = len(df)
    M = len(df.columns)
    indices = np.arange(N)
    rcol = [((1-0.3*i)/1, 0.0, 0.0, 1.0) for i in range(M)]
    gcol = [(0.0, 1/(M-i), 0.0, 1.0) for i in range(M)]
    (gcol if df.max().max() > 0 else rcol)
    for i, label, col in zip(range(M), (df.columns if df.max().max() > 0 else df.columns[::-1]), (gcol if df.max().max() > 0 else rcol)):
        plt.bar(indices, df[label], width=width, bottom=offset,
                alpha=alpha if i else 1, color=col, label=label)
        plt.xticks(indices, ['{}'.format(idx) for idx in df.index.values])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.045), ncol=M)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show:
        plt.show()
    if domain:
        plt.savefig(domain+'.pdf')
    return plt.gcf()


# Change the filename to match your data text file
filenames = ["Etest", "var90test", "cvar90test", "evar90test"]
methods = ["E", "VaR90", "CVaR90", "EVaR90"]
offset_var = "E"

# get DataFrame
plotmethods = [m for m in methods if m != offset_var]

dfs = [getdf(file) for file in filenames]
df = {key: value for key, value in zip(methods, dfs)}
domains = ['MR', 'GR', 'INV1', 'INV2', 'RS']
fullname = {"MR": "Machine Replacement (MR)", "GR": "Gamblers Ruin (GR)",
            "INV1": "Inventory Management (IM)",
            "INV2": "Inventory Management (IM2)", "RS": "River-swim (RS)"}

for domain in domains:
    offset_val = df["VaR90"][domain].max(
    ) + (df["VaR90"][domain].max() - df["CVaR90"][domain].max())*0.3

    data = [[df[m][("Method" if m == offset_var else domain)].values[i]
             for m in methods] for i in df[offset_var]["MR"].index]

    df1 = pd.DataFrame(
        data, columns=["Method"]+plotmethods).set_index("Method")
    df1 = df1 - offset_val

    overlapped_bar(df1, show=False, alpha=1, domain=domain, offset=offset_val,
                   title=fullname[domain], xlabel=' ', ylabel='Return')
    plt.clf()
