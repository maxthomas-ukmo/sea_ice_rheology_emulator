import xarray as xr
from pathlib import Path
import os
import random
import sys
import argparse
import concurrent.futures
import pickle

def make_feature_label_pairs(filename, feature_names, label_names, flatten=False):
    data = xr.open_dataset(filename)
    if flatten:
        data = data.stack(xy=('x', 'y'))
        data = data.dropna(dim='xy')
    features = data[feature_names]
    labels = data[label_names]
    pairs = []
    for itime in range(features.time_counter.size - 1):
        feature = features.isel(time_counter=itime)
        label = labels.isel(time_counter=itime+1)
        pairs.append((feature, label))
    return pairs

def make_args():
    parser = argparse.ArgumentParser(description='Make feature label pairs')
    parser.add_argument('filename', type=str, help='Filename of the data')
    parser.add_argument('feature_names', type=str, nargs='+', help='Feature names')
    parser.add_argument('label_names', type=str, nargs='+', help='Label names')
    parser.add_argument('save_path', type=str, help='Path to save the feature label pairs')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = make_args()
    pairs = make_feature_label_pairs(args.filename, args.feature_names, args.label_names)
    with open(args.save_path, 'wb') as f:
        pickle.dump(pairs, f)