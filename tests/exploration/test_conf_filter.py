import os
import numpy as np
import unittest, dpdata
from exploration.context import dpgen2
from dpgen2.utils.conf_filter import ConfFilter, ConfFilters

class OKFilter(ConfFilter):
    def check(
            self,
            conf : dpdata.System, 
    )-> bool:
        return True

class FailedFilter(ConfFilter):
    def check(
            self,
            conf : dpdata.System, 
    )-> bool:
        return False

class TestConfFilter(unittest.TestCase):
    def test_ok(self):
        ok0 = OKFilter()
        ok1 = OKFilter()
        ok2 = OKFilter()
        filters = ConfFilters()
        filters.add(ok0).add(ok1).add(ok2)
        self.assertTrue(filters.check(None))

    def test_failed(self):
        ok0 = OKFilter()
        ok1 = FailedFilter()
        ok2 = OKFilter()
        filters = ConfFilters()
        filters.add(ok0).add(ok1).add(ok2)
        self.assertFalse(filters.check(None))

