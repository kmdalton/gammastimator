import pandas as pd
import numpy as np
import argparse


def image_metadata(dataframe, keys = None):
    if keys is None:
        keys = [k for k in dataframe.keys() if 'ipm' in k.lower()]
        specifically_check = ['Io', 'BEAMX', 'BEAMY', 'Icryst']
        for k in specifically_check:
            if k in dataframe:
                keys.append(k)
    return dataframe[['RUN', 'IMAGENUMBER'] + list(keys)].groupby(['RUN','IMAGENUMBER']).mean()


def main():
    parser = argparse.parser()


if __name__=="__main__":
    main()
