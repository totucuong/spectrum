from spectrum.inferences.judge import Judge
import collections
import scipy.spatial.distance as dis
import numpy as np


class TruthFinder(Judge):
    """
    This class implement Truth Finder algorithm @yin2008truth
    """
    def __init__(self):
        super().__init__()

        # parameters set according to the paper
        self.__rho = 0.5
        self.__gamma = 0.3
        self.__max_iter = 10
        self.__sim_threshold = 0.1

    def fit(self, claims):
        """
        Parameters:
        -----------
        claims:    a list of Claims

        Returns
        -------
        None
        """
        self.index(claims)
