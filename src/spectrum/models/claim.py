__author__ = 'totucuong'
__date__ = '12/5/17'


class Claim:
    """
    This class represent a claim.
    """


    def __init__(self, subject=None, predicate=None, object=None, source=None):
        self.__subject = subject
        self.__predicate = predicate
        self.__object = object
        self.__source = source
        self.__confidence = 1

    def __str__(self):
        str = '[%s,%s,%s, by %s]' % (self.__subject, self.__predicate, self.__object, self.__source)
        return str

    @property
    def source(self):
        return self.__source

    @property
    def confidence(self):
        return self.__confidence

    @confidence.setter
    def confidence(self, confidence):
        self.__confidence = confidence

    @property
    def subject(self):
        return self.__subject

    @property
    def predicate(self):
        return self.__predicate

    @property
    def object(self):
        return self.__object

