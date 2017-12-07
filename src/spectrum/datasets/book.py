__author__ = 'totucuong'
__date__ = '12/5/17'

import pandas as pd
from zipfile import ZipFile
import ntpath

from spectrum.models.claim import Claim
from spectrum.datasets.dataset import DataSet


class Book(DataSet):
    """
    This class provides access to the books datasets.
    """

    def __init__(self):
        DataSet.__init__(self)
        self.data_path =  ntpath.dirname(__file__) + '/data/book.zip'
        self.__populate()

    def __populate(self):
        # read claims
        with ZipFile(self.data_path) as myzip:
            with myzip.open('book.txt') as book:
                df = pd.read_table(book, sep='\t', header=None,
                                   names=['source', 'book_id', 'book_name', 'authors'])
                self.claims = DataSet.parse(df.to_dict(orient='record'), sub_key='book_id', pred_key='authors', src_key='source', type='table')

            # read ground truths
            with myzip.open('book_golden.txt') as book_truth:
                df = pd.read_table(book_truth, sep='\t', header=None, names=['book_id', 'authors'])
                self.truths = DataSet.parse(df.to_dict(orient='record'), sub_key='book_id', pred_key='authors', type='table')

