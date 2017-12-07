__author__ = 'totucuong'
__date__ = '12/6/17'

from abc import abstractmethod
from spectrum.models.claim import Claim

class DataSet:

    def __init__(self):
        self.__claims = list()
        self.__truths = list()

    @staticmethod
    def parse(records, sub_key=None, pred_key=None, obj_key=None, src_key=None, type=None):
        """
        This method parse a list of records, i.e., dictionaries, into triples (subject, predicate, object).
        There are two kind of records: table record and triple records.

        Each triple record is already in the triple. The input keys will provide necessary mapping:
            subject = record[sub_key]
            predicate = record[pred_key]
            object = record[obj_key]

        Each table record is a row of a table. Hence:
            subject = record[sub_key]
            predicate = pred_key (not record[pred_key]), i.e., the header of a table column is the predicate
            object = record[pred_key]

        """
        if type is None:
            raise ValueError("type must be specified from {triple, table}")

        facts = []
        if type == 'table':
            predicate = pred_key
            if src_key is None:
                facts = [Claim(r[sub_key], predicate, r[pred_key]) for r in records]
            else:
                # with source info
                facts = [Claim(r[sub_key], predicate, r[pred_key], r[src_key]) for r in records]
        elif type == 'triple':
            if src_key is None:
                facts = [Claim(r[sub_key], r[pred_key], r[obj_key]) for r in records]
            else:
                facts = [Claim(r[sub_key], r[pred_key], r[obj_key], r[src_key]) for r in records]
        else:
            raise ValueError("%s type is not acceptable" % type)

        return facts

    @abstractmethod
    def __populate(self):
        pass

    @property
    def nclaims(self):
        return len(self.__claims)

    @property
    def ntruths(self):
        return len(self.__truths)

    @property
    def claims(self):
        return self.__claims

    @claims.setter
    def claims(self, v):
        self.__claims = v

    @property
    def truths(self):
        return self.__truths

    @truths.setter
    def truths(self, v):
        self.__truths = v


    def __str__(self):
        stats = dict()
        stats['#claims'] = self.nclaims
        stats['#truths'] = self.ntruths
        return stats.__str__()