# MNIST 神經網路專案

本專案實作了一個用於 MNIST 資料集的神經網路模型。它包含了資料載入、視覺化、模型訓練和預測的功能。

## 專案結構

- `mnist_loader.py`：下載、儲存和載入 MNIST 資料集。
- `mnist_show.py`：顯示 MNIST 圖片。
- `neuralnet_mnist.py`：實作神經網路推論過程。
- `train_network.py`：訓練神經網路模型並儲存訓練後的權重。
- `logic_functions.py`：包含必要的邏輯函數，例如 sigmoid 和 softmax。
- `data/`：包含下載的 MNIST 資料集。
- `models/`：包含訓練後的模型權重。
- `plots/`：包含生成的圖表（如果有）。

## 使用方法

### 下載 MNIST 資料集

首先，下載並儲存 MNIST 資料集：

```bash
python mnist_loader.py
```

此腳本將下載資料集並儲存到 `dataset/` 目錄。

### 訓練模型

要訓練神經網路模型：

```bash
python train_network.py
```

這將訓練模型並將訓練後的權重儲存到 `models/` 目錄。

### 進行預測

要對新資料進行預測：

```bash
python neuralnet_mnist.py
```

此腳本將載入訓練後的權重並執行預測。

### 顯示資料

要顯示 MNIST 圖片：

```bash
python mnist_show.py
```

此腳本將顯示 MNIST 圖片。

## 更多資訊

有關模型架構和實作的詳細資訊，請參考各個 Python 檔案中的原始碼。

## 需求

要執行此專案，請確保已安裝必要的函式庫。在終端機執行以下命令：

```bash
pip install -r requirements.txt
