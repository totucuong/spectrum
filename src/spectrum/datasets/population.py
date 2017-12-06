__author__ = 'totucuong'
__date__ = '12/5/17'

import pandas as pd
from zipfile import ZipFile
from spectrum.datasets.data import Data

class Population(Data):
    """
    This class provide access to population dataset
    """
    def __init__(self, data_home=None):
        Data.__init__(self, data_home)
        self.data_path = data_home + '/population.zip'
        self.__populate()

    def __populate(self):
        with ZipFile(self.data_path) as myzip:
            # read claims
            with myzip.open('population_claims.csv') as population_claims:
                df = pd.read_csv(population_claims, header=0)
                self.claims = self.parse(df.to_dict(orient='record'), sub_key='ObjectID', pred_key='PropertyID',
                                         obj_key='PropertyValue', src_key='SourceID', type='triple')

            # read ground truths
            with myzip.open('population_truth.csv') as population_truths:
                df = pd.read_csv(population_truths, header = 0)
                self.truths = self.parse(df.to_dict(orient='record'), sub_key='ObjectID', pred_key='PropertyID',
                                         obj_key='PropertyValue', type='triple')

# class Population(Data):
#     """
#     This class provide access to population dataset
#     """
#     def __init__(self, data_home=None):
#         self.data_path = data_home + '/population.zip'
#
#         # datasets structures
#         self.__claims = list()
#         self.__truths = list()
#
#         # populate claims and truths
#         self.__populate()
#
#     def __populate(self):
#         with ZipFile(self.data_path) as myzip:
#             # read claims
#             with myzip.open('population_claims.csv') as population_claims:
#                 df = pd.read_csv(population_claims, header=0)
#                 records = df.to_dict(orient='record')
#                 self.__claims = self.__parse(records)
#
#             # read ground truths
#             with myzip.open('population_truth.csv') as population_truths:
#                 df = pd.read_csv(population_truths, header = 0)
#                 records = df.to_dict(orient='record')
#                 self.__truths = self.__parse(records)
#
#     def __parse(self, records):
#         facts = [Claim(r['ObjectID'], r['PropertyID'], r['PropertyValue']) for r in records]
#         return facts
#
#     @property
#     def nclaims(self):
#         """number of claims"""
#         return len(self.__claims)
#
#     @property
#     def ntruths(self):
#         """number of truths"""
#         return len(self.__truths)
#
#     def get_claims(self):
#         return self.__claims
#
#     def get_truths(self):
#         return self.__truths
#
#     def __str__(self):
#         stats = dict()
#         stats['#claims'] = self.nclaims
#         stats['#truths'] = self.ntruths
#         return stats.__str__()
