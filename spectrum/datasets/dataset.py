from abc import abstractmethod
from spectrum.models.triple import Triple

class DataSet:

    def __init__(self):
        self.__claims = list()
        self.__truths = list()

    @staticmethod
    def parse(records, sub_key=None, pred_key=None, obj_key=None, src_key=None, conf_key=None, type=None):
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

        Parameters
        ----------
        records: a list of records. Each record is a dictionary.
        sub_key: key of the subject field
        pred_key: key of predicate field
        src_key: key of source field
        conf_key: key of confidence field
        type: type of record.
            'triple'
            'table'

        Returns
        -------
        a list of Claims
        """
        if type is None or type not in {'triple', 'table'}:
            raise ValueError("type must be specified from {triple, table}")

        facts = []
        if type == 'table':
            predicate = pred_key
            if src_key is None:
                facts = [Triple(r[sub_key], predicate, r[pred_key]) for r in records]
            else:
                # with source info
                facts = [Triple(r[sub_key], predicate, r[pred_key], r[src_key]) for r in records]
        else:
            if (src_key is None) and (conf_key is None) :
                facts = [Triple(r[sub_key], r[pred_key], r[obj_key]) for r in records]
            elif conf_key is None:
                facts = [Triple(r[sub_key], r[pred_key], r[obj_key], r[src_key]) for r in records]
            else:
                facts = [Triple(r[sub_key], r[pred_key], r[obj_key], r[src_key], r[conf_key]) for r in records]


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

class DefaultDataSet(DataSet):
    """
    Load claims/truths from files

    Parameters
    ----------
    claim_path: path to claims file
    truth_path: path to truth file
    """
    def __init__(self, claim_path=None, truth_path=None):
        super().__init__()
        if claim_path is None:
            raise ValueError('claim_path must be supplied')
        self.__claim_path = claim_path
        self.__truth_path = truth_path
        self.__populate()

    def __populate(self):
        self.claims = self.__read_from(self.__claim_path)
        self.truths = self.__read_from(self.__truth_path)

    def __read_from(self, file_path):
        facts = list()
        with open(file_path, 'r') as f:
            for l in f:
                fields = l.split(',')
                facts.append(Triple(*fields))
        return facts
