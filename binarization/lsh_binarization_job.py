import sys
import numpy as np
import faiss
import math
import argparse
import pandas as pd


class Initializer:

    def __init__(self, args):
        self.args = args
        self.long_long_size = 8
        self.long_size = 4
        self.int_size = 2
        self.output_file_binary_components = args.binary_components
        self.input_file_items_ids = args.items_ids
        if self.output_file_binary_components is None:
            print('One must provide [--binary_components], [--encoded_items] or [--index_file].'
                  + '\r\n' + 'Please use [-h] or [--help] for more information.')
            sys.exit(1)

        self.input_file_index = args.index_file
        self.input_file_encoded_items = args.encoded_items

        if self.input_file_encoded_items is None and self.input_file_index is None:
            print('One must provide [--encoded_items] or [--index_file].'
                  + '\r\n' + 'Please use [-h] or [--help] for more information.')
            sys.exit(1)

        self.component_type_size = args.component_type_size
        if self.component_type_size is self.long_long_size:
            self.data_type = np.ulonglong
        elif self.component_type_size is self.long_size:
            self.data_type = np.uint32
        elif self.component_type_size is self.int_size:
            self.data_type = np.uint16
        else:
            self.component_type_size = self.long_long_size
            self.data_type = np.ulonglong

        if self.input_file_encoded_items is not None:
            self.encoded_items_vec = np.load(self.input_file_encoded_items)
        else:
            self.encoded_items_vec = None

        self.input_file_training_set = args.training_set
        if self.input_file_training_set is None:
            self.training_set_vec = self.encoded_items_vec
        else:
            self.training_set_vec = np.load(self.input_file_training_set)

        self.number_of_bits = args.number_of_bits

        if self.input_file_index is None:
            _, self.dimensions = self.encoded_items_vec.shape

            if self.number_of_bits is None:
                self.number_of_bits = self.dimensions * 2

            self.lsh_index = self.get_lsh_index()
        else:
            self.lsh_index = faiss.read_index(self.input_file_index)
            self.dimensions = None

    def get_lsh_index(self):
        index = faiss.IndexLSH(self.dimensions, self.number_of_bits)
        index.train(self.training_set_vec)
        index.add(self.encoded_items_vec)
        return index


class Binarizer:

    def __init__(self, initializer):
        self.initializer = initializer

    def __get_binary_components(self):
        print('Type size: ' + str(self.initializer.component_type_size) + ' bytes.')
        bytes_per_vector = self.initializer.lsh_index.bytes_per_vec
        print('Bytes per vector: ' + str(bytes_per_vector))
        print('Number of bits: ' + str(bytes_per_vector * 8))
        vectors_bytes = faiss.vector_to_array(self.initializer.lsh_index.codes)
        n_components = math.ceil(bytes_per_vector / self.initializer.component_type_size)
        print('Number of binary components per vector: ' + str(n_components))

        n = int(len(vectors_bytes) / bytes_per_vector)
        resulting_components = np.zeros((n, n_components), dtype=self.initializer.data_type)

        vector_start = 0
        vector_stop = bytes_per_vector
        vector_count = 0
        while vector_count < n:
            binary_vector = np.zeros((n_components,), dtype=self.initializer.data_type)
            vector_bytes = vectors_bytes[vector_start:vector_stop]
            component_start = 0
            component_stop = self.initializer.component_type_size
            component_count = 0
            while component_count < n_components:
                numerical_component_bytes = vector_bytes[component_start:component_stop]
                numerical_component = int.from_bytes(numerical_component_bytes, byteorder='big', signed=False)
                binary_vector[component_count] = numerical_component
                component_start = component_stop
                component_stop = component_stop + self.initializer.component_type_size
                component_count += 1
            resulting_components[vector_count] = binary_vector
            vector_start = vector_stop
            vector_stop = vector_stop + bytes_per_vector
            vector_count += 1
        print('Total items: ' + str(len(resulting_components)))
        return resulting_components

    def __zip_with_ids(self, components):
        ids = np.load(self.initializer.input_file_items_ids)
        id_series = pd.Series(ids, name='item_id')
        bin_vec_series = pd.Series(components.tolist(), name='binary_vector')
        frame = {'item_id': id_series, 'binary_vector': bin_vec_series}
        df = pd.DataFrame(frame)
        return df

    def save_binary_components(self):
        components = self.__get_binary_components()
        if self.initializer.input_file_items_ids is not None:
            result = self.__zip_with_ids(components)
        else:
            bin_vec_series = pd.Series(components.tolist(), name='binary_vector')
            frame = {'binary_vector': bin_vec_series}
            result = pd.DataFrame(frame)
        result.to_csv(self.initializer.output_file_binary_components + '.csv', index=False)
        print('Result is stored in: ' + self.initializer.output_file_binary_components + '.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoded_items', help='The actual dataset (vectors) in .npy format that will be converted '
                                                'to binary representation, if not specified one must provide trained '
                                                'and populated Faiss LSH index file [--index_file]. [--encoded_items] '
                                                'argument will be ignored if trained and populated Faiss LSH index '
                                                'file is provided [--index_file].', required=False)
    parser.add_argument('--training_set', help='This dataset in .npy format will be used to train LSH index, if not '
                                               'specified the actual dataset will be used as a training set. This '
                                               'argument will be ignored if trained and populated Faiss LSH index '
                                               'file is provided [--index_file].', required=False)
    parser.add_argument('--index_file', help='Trained and populated Faiss LSH index file including full path, if not '
                                             'specified one must provide the actual dataset [--encoded_items].',
                        required=False)
    parser.add_argument('--binary_components', help='The name of resulting file in .csv format including full path.',
                        required=False)
    parser.add_argument('--component_type_size', help='The size of a single binary component in bytes, the default '
                                                      'value is unsigned LONG LONG which is 8 (bytes).', required=False,
                        type=int)
    parser.add_argument('--number_of_bits',
                        help='The number of bits that will be used in LSH index, if not specified d '
                             '* 2 where d is the number of dimensions in the actual dataset will be '
                             'used as a default value. This argument will be ignored if trained and '
                             'populated Faiss LSH index file is provided [--index_file].', required=False, type=int)
    parser.add_argument('--items_ids', help='The name of file in .npy format including full path that contains a set '
                                            'of identifiers. If specified, the set will be zipped with elements in '
                                            'resulting output .csv file with the order provided.', required=False)

    Binarizer(
        Initializer(parser.parse_args())
    ).save_binary_components()
