# MNIST 神經網路專案

本專案實作了一個用於 MNIST 資料集的三層神經網路模型。它包含了資料載入、視覺化、模型訓練和預測的功能。

## 專案結構

- `dataset/`: 包含下載的 MNIST 資料集 (`.npz` 格式)。
- `mnist_loader.py`: 下載、儲存和載入 MNIST 資料集。  此檔案負責處理 MNIST 資料集的下載和載入，並將其轉換為適合神經網路模型的格式。
- `mnist.py`: 提供載入 MNIST 資料集的函數。
- `mnist_show.py`: 顯示 MNIST 圖片。  此檔案提供視覺化 MNIST 資料集圖片的功能。
- `logic_functions.py`: 包含基本激活函數 (sigmoid, ReLU, softmax, step function) 和加權和計算。
- `logic_gates.py`: 實作邏輯閘 (AND, NAND, OR, XOR)。
- `perceptron.py`: 實作感知器。
- `three_layer_neural_network.py`: 實作三層神經網路。
- `neuralnet_mnist.py`: 執行神經網路推論。  此檔案負責載入訓練好的模型，並對新的輸入資料進行預測。
- `train_network.py`: 訓練神經網路模型並儲存訓練後的權重。  此檔案負責訓練神經網路模型，並將訓練後的權重儲存到 `models/` 目錄。
- `models/`: 儲存訓練好的神經網路模型權重。
- `plots/`: 儲存生成的圖表（如果有）。
- `requirements.txt`: 列出專案所需的 Python 函式庫。
- `main.py`: (如果存在) 主要執行程式。
- `test_dataset/`: (如果存在) 測試資料集。
- `tests/`: (如果存在) 單元測試程式碼。


## 模型架構

本專案使用三層神經網路，包含輸入層、隱藏層和輸出層。輸入層接收 MNIST 圖片的像素值，隱藏層使用 ReLU 激活函數，輸出層使用 softmax 激活函數。

## 資料集格式

MNIST 資料集儲存在 `dataset/mnist.npz` 檔案中，包含訓練集圖片和標籤，以及測試集圖片和標籤。

## 使用方法

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 下載 MNIST 資料集

```bash
python mnist_loader.py
```

### 訓練模型

```bash
python train_network.py
```

### 進行預測

```bash
python neuralnet_mnist.py
```

### 顯示資料

```bash
python mnist_show.py
```

## 更多資訊

有關模型架構和實作的詳細資訊，請參考各個 Python 檔案中的原始碼。

## 需求

要執行此專案，請確保已安裝必要的函式庫。
