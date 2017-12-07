from zipfile import ZipFile
from spectrum.datasets.dataset import DataSet
import ntpath
import pandas as pd

class Population(DataSet):
    """
    This class provide access to population dataset
    """
    def __init__(self):
        self.data_path = ntpath.dirname(__file__) + '/data/population.zip'
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
