# PyTorch 手写数字识别

这个示例使用 PyTorch 和 torchvision 在 MNIST 数据集上训练一个简单 CNN，用于识别 0-9 的手写数字。

## 安装依赖

```powershell
pip install -r requirements.txt
```

如果需要 GPU 版本 PyTorch，请按你的 CUDA 版本参考 PyTorch 官方安装命令。

## 训练模型

```powershell
python mnist_pytorch.py train --epochs 3
```

脚本会自动下载 MNIST 数据集到 `data/`，训练完成后把模型保存到 `models/mnist_cnn.pt`。

## 预测单张图片

```powershell
python mnist_pytorch.py predict --image-path path\to\digit.png
```

也可以显式指定模型路径：

```powershell
python mnist_pytorch.py predict --model-path models\mnist_cnn.pt --image-path path\to\digit.png
```

## 可视化预测结果

```powershell
python mnist_pytorch.py visualize --model-path models\mnist_cnn.pt --image-path path\to\digit.png
```

默认会把可视化结果保存到 `outputs/prediction.png`，图片中包含输入数字、预测类别、置信度和 0-9 的概率柱状图。

也可以指定输出路径：

```powershell
python mnist_pytorch.py visualize --model-path models\mnist_cnn.pt --image-path path\to\digit.png --output-path outputs\my_prediction.png
```

如果希望弹出窗口展示，追加 `--show`：

```powershell
python mnist_pytorch.py visualize --model-path models\mnist_cnn.pt --image-path path\to\digit.png --show
```

## 常用参数

- `--epochs`：训练轮数，默认 `3`
- `--batch-size`：批大小，默认 `64`
- `--lr`：学习率，默认 `0.001`
- `--data-dir`：数据集目录，默认 `data`
- `--model-path`：模型保存或加载路径，默认 `models/mnist_cnn.pt`
