# MNIST 手寫數字分類器

本專案使用多層感知器神經網路，對 MNIST 資料集中的手寫數字進行分類。它包含資料載入、模型訓練、預測和評估功能。

## 專案結構

- `dataset/`: 包含已下載的 MNIST 資料集 (`.npz` 格式)。
- `mnist_loader.py`: 下載、儲存和載入 MNIST 資料集。
- `mnist_show.py`: 顯示 MNIST 圖片。
- `logic_gates.py`: 實作邏輯閘。
- `matrix_multiplication_example.py`: 矩陣乘法範例。
- `multi_dimensional_array_example.py`: 多維陣列範例。
- `neuralnet_mnist.py`: 載入訓練好的模型，對測試集進行預測，並計算準確度。
- `perceptron.py`: 實作感知器。
- `plot_utils.py`: 繪圖工具。
- `README.md`: 您現在正在閱讀的文件。
- `requirements.txt`: 列出所需的 Python 函式庫。
- `sample_weight.pkl`: 儲存訓練好的神經網路模型權重。
- `setup.py`: 專案設置檔案。
- `simple_net.py`: 簡單的神經網路實作。
- `three_layer_neural_network.py`: 實作三層神經網路架構。
- `train_network.py`: 訓練神經網路模型並儲存訓練好的權重。
- `train_neuralnet.py`: 訓練神經網路模型的另一個版本。
- `two_layer_net.py`: 實作兩層神經網路。
- `two_layer_neural_network.py`: 實作兩層神經網路的另一個版本。
- `common/`: 包含共用的函數和層。
  - `__init__.py`: 初始化檔案。
  - `functions.py`: 激活函數。
  - `layers.py`: 層的實作。
  - `multi_layer_net.py`: 多層神經網路的實作。
  - `multi_layer_net_deep.py`: 深層多層神經網路的實作。
  - `multi_layer_net_bn.py`: 具有 Batch Normalization 的多層神經網路。
  - `update.py`: 更新規則，包含各種優化器如 SGD、Momentum、AdaGrad 等。
  - `util.py`: 輔助工具，提供數據處理、模型評估等功能。
- `models/`: 儲存訓練好的神經網路模型權重。
- `plots/`: 儲存生成的圖表。
  - `activation_histograms.py`: 激活函數直方圖。
  - `batch_norm_test.py`: Batch Normalization 測試。
  - `optimizer_compare_mnist.py`: 優化器比較 (MNIST)。
  - `optimizer_compare.py`: 優化器比較。
  - `plot_gradient_descent.py`: 繪製梯度下降過程。
  - `plot_gradient.py`: 繪製梯度。
  - `plot_numerical_derivative.py`: 繪製數值微分。
  - `plot_relu_function.py`: 繪製 ReLU 函數。
  - `plot_sigmoid_function.py`: 繪製 Sigmoid 函數。
  - `plot_sine_wave.py`: 繪製正弦波。
  - `plot_step_function.py`: 繪製階梯函數。
  - `plot.py`: 通用繪圖腳本。
  - `softmax_plot.py`: 繪製 softmax 函數。
  - `weight_init_activation_histogram.py`: 權重初始化激活直方圖。
  - `weight_init_bn_compare.py`: 權重初始化與 Batch Normalization 比較。
  - `weight_init_compare_deep.py`: 深層網路權重初始化比較。
  - `weight_init_compare.py`: 權重初始化比較。
- `test_dataset/`: 測試資料集。
- `tests/`: 單元測試程式碼。
  - `__init__.py`: 初始化檔案。
  - `test_all.py`: 所有測試的入口。
  - `test_batch_norm.py`: Batch Normalization 測試。
  - `test_gates.py`: 邏輯閘測試。
  - `test_install.py`: 安裝測試。
  - `test_logic_functions.py`: 邏輯函數測試。
  - `test_mnist.py`: MNIST 相關測試。
  - `test_multi_layer_net.py`: 多層神經網路測試。
  - `test_simple_net.py`: 簡單神經網路測試。
  - `test_two_layer_net.py`: 兩層神經網路測試。
- `utils/`: 輔助函數。

## 模型架構

本專案使用三層神經網路。輸入層接收 MNIST 圖片的像素值。隱藏層使用 sigmoid 激活函數，輸出層使用 softmax 激活函數進行多類別分類。

### 多層神經網路類別

- **MultiLayerNet**: 基本的多層神經網路，包含輸入層、隱藏層和輸出層。
- **MultiLayerNetDeep**: 深層多層神經網路，允許更多隱藏層。
- **MultiLayerNetBN**: 具有 Batch Normalization 的多層神經網路，旨在加速訓練並減少過擬合。

## 資料集格式

MNIST 資料集儲存在 `dataset/mnist.npz` 檔案中，包含訓練圖片和標籤，以及測試圖片和標籤。

## 使用方式

### 前置條件

- Python 3.x
- NumPy
- TensorFlow/Keras
- 其他必要的函式庫 (列在 `requirements.txt` 中)

### 安裝

```bash
pip install -r requirements.txt
```

### 訓練模型

```bash
python train_network.py
```

### 進行預測

```bash
python neuralnet_mnist.py
```

## 更多資訊

詳細資訊請參考各個 Python 檔案。

## 需求

請確保所有必要的函式庫已安裝，才能執行此專案。

## 貢獻

如果您想貢獻此專案，請遵循以下步驟：

1. Fork 此專案。
2. 創建您的功能分支 (`git checkout -b feature/YourFeature`).
3. 提交您的更改 (`git commit -m 'Add some feature'`).
4. 推送至分支 (`git push origin feature/YourFeature`).
5. 創建一個 Pull Request。

## 許可證

本專案使用 MIT 許可證。請參閱 [LICENSE](LICENSE) 文件以獲取更多資訊。

## 致謝

- 感謝 [TensorFlow](https://www.tensorflow.org/) 提供的深度學習框架。
- 感謝 [NumPy](https://numpy.org/) 提供的數值計算庫。
- 感謝 [MNIST](http://yann.lecun.com/exdb/mnist/) 資料集的提供者。
