import unittest
import sys
import os

sys.path.insert(0, os.path.split(os.path.split(__file__)[0])[0])
import lighthouse as lh

class TestInstall(unittest.TestCase):
    def test_home_variable(self):
        self.assertTrue("LightHouse_Home" in os.environ, "Need to set up LightHouse_HOME environment variable")
