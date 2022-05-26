import pandas as pd
import os
import tensorflow as tf
import numpy as np

class DataLoader():
    def arrange_data(self, features, labels, names):
        # input file should be in dataframe type
                                       
        f_array = []
        l_array = []

        for name in names:

            f_vector = list(features.loc[name])
            l = labels.loc[name]

            f_array.append(f_vector)
            l_array.append(l)

        f_array = np.array(f_array)
        l_array = np.array(l_array)
        
        return f_array, l_array
    
    def make_dataset(self, features, labels,
            batch_size=128, repeat=False, shuffle=True, buffer_size=20000):

        f_vs = tf.data.Dataset.from_generator(
                lambda: (f for f in features),
                output_types=tf.float32,
                )

        f_vs = f_vs.cache()
            
        ys = labels.copy()
        if ys is not None:
            if len(ys.shape) == 1:
                ys = np.reshape(ys, (-1, 1))

            _ys = ys
            ys = tf.data.Dataset.from_generator(
                    lambda: (y for y in _ys),
                    output_types=tf.float32,
                    )
            dataset = tf.data.Dataset.zip((f_vs, ys))
        else:
            dataset = f_vs

        if shuffle:
            dataset = dataset.shuffle(buffer_size)
            
        if repeat:
            dataset = dataset.repeat()
        
        dataset = dataset.batch(batch_size)

        return dataset


