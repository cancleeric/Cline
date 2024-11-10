# MNIST 手寫數字分類器

本專案使用多層感知器神經網路，對 MNIST 資料集中的手寫數字進行分類。它包含資料載入、模型訓練、預測和評估功能。

## 專案結構

- `dataset/`: 包含已下載的 MNIST 資料集 (`.npz` 格式)。
- `mnist_loader.py`: 下載、儲存和載入 MNIST 資料集。
- `mnist.py`: (如果存在) 與 MNIST 資料集相關的額外函數。
- `logic_functions.py`: 包含基本激活函數 (sigmoid、ReLU、softmax、階梯函數) 和加權和計算。
- `three_layer_neural_network.py`: 實作三層神經網路架構。
- `train_network.py`: 訓練神經網路模型並儲存訓練好的權重。
- `neuralnet_mnist.py`: 載入訓練好的模型，對測試集進行預測，並計算準確度。
- `models/`: 儲存訓練好的神經網路模型權重 (預設為 `sample_weight.pkl`)。
- `plots/`: 儲存生成的圖表 (如果有)。
- `requirements.txt`: 列出所需的 Python 函式庫。
- `main.py`: (如果存在) 主要執行腳本。
- `test_dataset/`: (如果存在) 測試資料集。
- `tests/`: (如果存在) 單元測試程式碼。
- `utils/`: (如果存在) 輔助函數。


## 模型架構

本專案使用三層神經網路。輸入層接收 MNIST 圖片的像素值。隱藏層使用 sigmoid 激活函數，輸出層使用 softmax 激活函數進行多類別分類。

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
