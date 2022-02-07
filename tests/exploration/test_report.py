from context import dpgen2
import os, textwrap
import numpy as np
import unittest
from collections import Counter
from dpgen2.exploration.report import NaiveExplorationReport

class TestNaiveExplorationReport(unittest.TestCase):
    def test_naive_fv(self):
        counter_f = Counter()
        counter_v = Counter()
        counter_f['candidate'] = 5
        counter_f['accurate'] = 4
        counter_f['failed'] = 1
        counter_v['candidate'] = 7
        counter_v['accurate'] = 2
        counter_v['failed'] = 1
        report = NaiveExplorationReport(counter_f, counter_v)
        self.assertAlmostEqual(report.ratio('force', 'candidate'), 0.5)
        self.assertAlmostEqual(report.ratio('force', 'accurate'), 0.4)
        self.assertAlmostEqual(report.ratio('force', 'failed'), 0.1)
        self.assertAlmostEqual(report.ratio('virial', 'candidate'), 0.7)
        self.assertAlmostEqual(report.ratio('virial', 'accurate'), 0.2)
        self.assertAlmostEqual(report.ratio('virial', 'failed'), 0.1)
        self.assertAlmostEqual(report.candidate_ratio(), 0.5)
        self.assertAlmostEqual(report.accurate_ratio(), 0.4)
        self.assertAlmostEqual(report.failed_ratio(), 0.1)
        

    def test_naive_f(self):
        counter_f = Counter()
        counter_v = Counter()
        counter_f['candidate'] = 5
        counter_f['accurate'] = 4
        counter_f['failed'] = 1
        counter_v['candidate'] = 0
        counter_v['accurate'] = 0
        counter_v['failed'] = 0
        report = NaiveExplorationReport(counter_f, counter_v)
        self.assertAlmostEqual(report.ratio('force', 'candidate'), 0.5)
        self.assertAlmostEqual(report.ratio('force', 'accurate'), 0.4)
        self.assertAlmostEqual(report.ratio('force', 'failed'), 0.1)
        self.assertAlmostEqual(report.ratio('virial', 'candidate'), None)
        self.assertAlmostEqual(report.ratio('virial', 'accurate'), None)
        self.assertAlmostEqual(report.ratio('virial', 'failed'), None)
        self.assertAlmostEqual(report.candidate_ratio(), 0.5)
        self.assertAlmostEqual(report.accurate_ratio(), 0.4)
        self.assertAlmostEqual(report.failed_ratio(), 0.1)


    def test_naive_failed(self):
        counter_f = Counter()
        counter_v = Counter()
        counter_f['candidate'] = 5
        counter_f['accurate'] = 4
        counter_f['failed'] = 1
        counter_v['candidate'] = 7
        counter_v['accurate'] = 2
        counter_v['failed'] = 1
        report = NaiveExplorationReport(counter_f, counter_v)
        with self.assertRaises(RuntimeError) as context:
            report.ratio('foo', 'candidate')
        self.assertTrue('invalid quantity foo' in str(context.exception))
        with self.assertRaises(RuntimeError) as context:
            report.ratio('force', 'bar')
        self.assertTrue('invalid item bar' in str(context.exception))
