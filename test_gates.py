import unittest
import numpy as np
from logic_gates import XORGate, ANDGate, NANDGate, ORGate

class TestLogicGates(unittest.TestCase):

    def test_and_gate(self):
        and_gate = ANDGate()
        self.assertEqual(and_gate.calculate(0, 0), 0)
        self.assertEqual(and_gate.calculate(0, 1), 0)
        self.assertEqual(and_gate.calculate(1, 0), 0)
        self.assertEqual(and_gate.calculate(1, 1), 1)

    def test_nand_gate(self):
        nand_gate = NANDGate()
        self.assertEqual(nand_gate.calculate(0, 0), 1)
        self.assertEqual(nand_gate.calculate(0, 1), 1)
        self.assertEqual(nand_gate.calculate(1, 0), 1)
        self.assertEqual(nand_gate.calculate(1, 1), 0)

    def test_or_gate(self):
        or_gate = ORGate()
        self.assertEqual(or_gate.calculate(0, 0), 0)
        self.assertEqual(or_gate.calculate(0, 1), 1)
        self.assertEqual(or_gate.calculate(1, 0), 1)
        self.assertEqual(or_gate.calculate(1, 1), 1)

    def test_xor_gate(self):
        xor_gate = XORGate()
        self.assertEqual(xor_gate.calculate(0, 0), 0)
        self.assertEqual(xor_gate.calculate(0, 1), 1)
        self.assertEqual(xor_gate.calculate(1, 0), 1)
        self.assertEqual(xor_gate.calculate(1, 1), 0)

if __name__ == '__main__':
    unittest.main()
