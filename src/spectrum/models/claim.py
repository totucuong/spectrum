__author__ = 'totucuong'
__date__ = '12/5/17'


class Claim:
    """
    This class represent a claim.
    """


    def __init__(self, subject='', predicate='', object=''):
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.confidence = 1
        self.source = ''

    def __str__(self):
        str = '[%s,%s,%s]' % (self.subject, self.predicate, self.object)
        return str





