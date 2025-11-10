#! /bin/python
# gvim: set fileencoding=utf-8
#
# (c) September 2, 2025, José Martinez, Polytech Nantes, University of Nantes
#
# Licence:  proprietary
# The use of this library is not authorised outside the Polytechnic School of the University of Nantes.
#


"""
This module provides the translation of the relational schema of the game into classes.
Notice that the IDs of the relational schema are pointers to class instances.

An instance of the relational schema is split into:

    * permanents facts, named "World," and
    * changing situations, stored in "State," but also referencing the world.

This module is dedicated to the "World," i.e., the description of the whole relational schema but 'State'.
"""


__all__ = [
    'Adjacencies',
    'RegionAdjacencies',
    'World',
    ]


from typing import List, Dict, Tuple, Set
# import strategy.strategy as S # Strategy contains circular imports into the model
from model.region import Region, RegionName, Regions
from model.territory import Territory, TerritoryName, Territories
from model.army import Army, Colour, Armies, ARMY_COLOURS


Adjacencies = Set[Tuple[Territory, Territory]]
"""
Adjacencies is the relational way to represent edges in a graph.
"""


RegionAdjacencies = Set[Tuple[Region, Region]]
"""
Adjacencies can be extended from territories to regions.
Two bordering territories from distinct regions imply that their regions are adjacent too.
"""


class World:
    """
    The so-called world consist of all the information that remain unchanged during a game.
    It is everything except the class 'State'.
    """

    def __init__ (self,
                  G:      List['Strategy'],
                  M:      Dict[RegionName, Set[TerritoryName]],
                  Gamma:  Dict[TerritoryName, Set[TerritoryName]]):
        assert len(G) >= 1, "at least one army, as a limit case"
        assert len({ t
                     for r in M
                     for t in M[r] }) >= len(G), "at least as many territories as armies"

        A = { c: Army(c, g)
              for (c, g) in zip(ARMY_COLOURS, G) }
        R = { r: Region(r, (n := len(M[r])) // 2 + n % 2)
              for r in M }
        T = { t: Territory(t, R[r])
              for r in M
             for t in M[r] }
        D = { (T[t1], T[t2])
              for t1 in Gamma
              for t2 in Gamma[t1]
              if t1 != t2 } | { (T[t2], T[t1])
                                for t1 in Gamma
                                for t2 in Gamma[t1]
                                if t1 != t2 } # adjacencies ensured irreflexive and symmetrical
        self._armies_index = A
        self._armies = { A[a] for a in A }
        self._regions_index = R
        self._regions = { R[r] for r in R }
        self._territories_index = T
        self._territories = { T[t] for t in T }
        self._adjacencies = D
        self._region_adjacencies = { (r1, r2)
                                      for t1 in T
                                      for t2 in T
                                      if t1 != t2
                                      if (r1 := T[t1].region) != (r2 := T[t2].region)}
        self._region_territories = { R[r]: { T[t]
                                              for t in T
                                              if T[t].region == r }
                                      for r in R }

    @property
    def armies (self) -> Armies:
        """
        From three to six armies fighting against each other on the board.
        """
        return self._armies

    @property
    def regions (self) -> Regions:
        """
        The regions of the used board.
        """
        return self._regions

    @property
    def territories (self) -> Territories:
        """
        The territories of the used board.
        """
        return self._territories

    @property
    def adjacencies (self) -> Adjacencies:
        """
        The borders between territories as pairs of territories.
        """
        return self._adjacencies

    @property
    def region_adjacencies (self) -> RegionAdjacencies:
        """
        The quotient set of territory adjacencies to their including regions.
        """
        return self._region_adjacencies

    @property
    def region_territories (self) -> Dict[Region, Set[Territory]]:
        """
        This is the regional view of the world.
        To each region are associated all of its territories.
        """
        return self._region_territories

    def army (self,
              a: Colour) -> Army:
        """
        Each army is externally identified by its colour.
        This function allows to retrieve it, i.e., its 'ID', based on its colour.
        """
        assert isinstance(a, Colour)
        assert a in self._armies_index

        return self._armies_index[a]

    def region (self,
                r: RegionName) -> Region:
        """
        Each region is externally identified by its name.
        This function allows to retrieve it, i.e., its 'ID', based on its name.
        """
        assert isinstance(r, RegionName)
        assert r in self._regions_index

        return self._regions_index[r]

    def territory (self,
                   t: TerritoryName) -> Territory:
        """
        Each territory is externally identified by its name.
        This function allows to retrieve it, i.e., its 'ID', based on its name.
        """
        assert isinstance(t, TerritoryName)
        assert t in self._territories_index

        return self._territories_index[t]
