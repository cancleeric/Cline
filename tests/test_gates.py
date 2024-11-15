import unittest
import numpy as np
import sys
import os

# 獲取當前腳本的絕對路徑，並將上一層資料夾添加到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


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
