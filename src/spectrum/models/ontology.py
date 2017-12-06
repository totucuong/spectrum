__author__ = 'totucuong'
__date__ = '12/6/17'

class Entity:
    """
    This class model an entity in the world.

    An entity can have multiple representations in the world. For example, the book 'Computing Essentials' by Daniel
     O'Leary and Timothy J O'Leary is listed under different names by publishers. Some examples are:
        1. Computing Essentials
        2. Computing Essentials 2007, Complete Edition
        3. Computing Essentials 2007

    Each entity with have an unique id. For books a good candidate would be its ISBN.

    Each entity will have multiple representations.
    """

    def __init__(self):
        self.id = None
        self.repr = None