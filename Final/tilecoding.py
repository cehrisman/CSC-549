import math
from six.moves import zip_longest

basehash = hash


class Tiling:
    def __init__(self, size):
        self.size = size
        self.overfullCount = 0
        self.dict = {}

    def count(self):
        return len(self.dict)

    def fullp(self):
        return len(self.dict) >= self.size

    def get_idx(self, obj):
        d = self.dict
        if obj in d:
            return d[obj]
        size = self.size
        count = self.count()
        if count >= size:
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coords, m):
    if isinstance(m, Tiling):
        return m.get_idx(tuple(coords))
    if isinstance(m, int):
        return basehash(tuple(coords)) % m
    if m is None:
        return coords

def tiles(ihtORsize, numtilings, floats, ints=[]):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [math.floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // numtilings)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize))
    return Tiles

def tileswrap(ihtORsize, numtilings, floats, wrapwidths, ints=None, readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    if ints is None:
        ints = []
    qfloats = [math.floor(f * numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b % numtilings) // numtilings
            coords.append(c % width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles