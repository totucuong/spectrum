__author__ = 'totucuong'
__date__ = '12/5/17'

import pandas as pd
from zipfile import ZipFile

from spectrum.models.claim import Claim
from spectrum.datasets.data import Data


class Book(Data):
    """
    This class provides access to the books datasets.
    """

    def __init__(self, data_home=None):
        Data.__init__(self, data_home)
        self.data_path = self.data_home + '/book.zip'
        self.__populate()

    def __populate(self):
        # read claims
        with ZipFile(self.data_path) as myzip:
            with myzip.open('book.txt') as book:
                df = pd.read_table(book, sep='\t', header=None,
                                   names=['source', 'book_id', 'book_name', 'authors'])
                self.claims = Data.parse(df.to_dict(orient='record'), sub_key='book_id', pred_key='authors', src_key='source', type='table')

            # read ground truths
            with myzip.open('book_golden.txt') as book_truth:
                df = pd.read_table(book_truth, sep='\t', header=None, names=['book_id', 'authors'])
                self.truths = Data.parse(df.to_dict(orient='record'), sub_key='book_id', pred_key='authors', type='table')

