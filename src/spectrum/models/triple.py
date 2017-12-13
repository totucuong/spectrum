class Triple:
    """
    This class represent a claim.
    """
    def __init__(self, *args):
        if len(args) < 3 or len(args) > 5:
            raise ValueError('Claim must be initialized by at least three and at most 5 arguments')
        if len(args) == 3:
            self.__subject = args[0]
            self.__predicate = args[1]
            self.__object = args[2]
            self.__source = ''
            self.confidence = 0.0
        elif len(args) == 4:
            self.__subject = args[0]
            self.__predicate = args[1]
            self.__object = args[2]
            self.__source = args[3]
            self.__confidence = 0.0
        elif len(args) == 5:
            self.__subject = args[0]
            self.__predicate = args[1]
            self.__object = args[2]
            self.__source = args[3]
            self.__confidence = float(args[4])


    def __str__(self):
        str = '%s|%s|%s|%s|%f' % (self.__subject, self.__predicate, self.__object, self.__source, self.__confidence)
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

    @property
    def entity(self):
        """
        data item in truth finding literature, which is the first two of a triple: (subject,predicate)
        """
        return '%s|%s' % (self.subject, self.predicate)

    @property
    def fact(self):
        """
        return a string representation of the fact provided by this claim
        """
        return '%s|%s|%s' % (self.__subject, self.__predicate, self.__object)


    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        return (self.subject == other.subject and self.predicate == other.predicate and self.object == other.object)