import numpy as np
import pandas as pd

DIRECTORY = '../statistics/scaler/'
if not DIRECTORY.endswith('/'):
    DIRECTORY += '/'


class Scaler(object):
    def __init__(self, file_name='state_action_reinterpreted'):
        mean_std = pd.read_csv(DIRECTORY + file_name + '.csv', index_col=0).drop(columns=['dimension'])
        self.offset = mean_std['mean'].values
        epsilon = max(min(mean_std['std'].min(), .1), 1e-4)
        self.scale = 1. / (mean_std['std'].values + epsilon) / 3.
        del mean_std

    def get(self):
        return self.offset, self.scale


def main():
    scaler = Scaler('state_action_reinterpreted')
    offset, scale = scaler.get()
    print('Offset: \n', offset[80:90])
    print('Scale: \n', scale[80:90])


if __name__ == '__main__':
    main()
