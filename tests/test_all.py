import unittest

# 匯入現有的測試模組
from . import test_mnist
import test_install
import test_gates

if __name__ == "__main__":
    # 創建測試套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 加載測試模組中的測試
    suite.addTests(loader.loadTestsFromModule(test_mnist))
    suite.addTests(loader.loadTestsFromModule(test_install))
    suite.addTests(loader.loadTestsFromModule(test_gates))

    # 運行測試
    runner = unittest.TextTestRunner()
    runner.run(suite)
