__author__ = 'totucuong'
__date__ = '12/5/17'

import pandas as pd
import collections
import pprint

from spectrum.models.claim import Claim

class Population:
    """
    This class provide access to population dataset
    """
    def __init__(self, path_to_claims='../../../data/population/claims/population_claims.csv',
                 path_to_ground_truth='../../../data/population/ground_truth/population_truth.csv'):
        # data paths
        self.path_to_claims = path_to_claims
        self.path_to_ground_truth = path_to_ground_truth

        # data structures
        self.subjects = set()
        self.predicates = set()
        self.claims = list()

        # populate data structures
        self.__populate_claims()

    def __populate_claims(self):
        df = pd.read_csv(self.path_to_claims, header=0)
        records = df.to_dict(orient='record')
        self.__parse(records)

    def __parse(self, records):
        for r in records:
            self.subjects.add(r['ObjectID'])
            self.predicates.add(r['PropertyID'])

            c = Claim()
            c.subject = r['ObjectID']
            c.predicate = r['PropertyID']
            c.object = r['PropertyValue']
            self.claims.append(c)

    @property
    def nclaims(self):
        return len(self.claims)

    def __str__(self):
        stats = dict()
        stats['#claims'] = self.nclaims
        stats['#subjects'] = len(self.subjects)
        stats['#predicates'] = len(self.predicates)
        return stats.__str__()


    def __read_ground_truth(self):
        pass


def main():
    population = Population()
    print(population)


if __name__ == "__main__":
    main()