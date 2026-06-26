# mini-model-arch — 模型架构 (C 语言实现)

深度学习模型架构的纯 C99 实现，涵盖 CNN、RNN/LSTM、Transformer、GAN 和 Diffusion 五大类。

## 模块

| 头文件 | 内容 |
|--------|------|
| `cnn_models.h` | LeNet-5, VGG, ResNet, Inception, Depthwise Separable Conv |
| `rnn_lstm.h` | RNN, LSTM, GRU, Bidirectional, Stacked RNN |
| `transformer_arch.h` | Self-Attention, Multi-Head, Encoder/Decoder, GPT, BERT |
| `gan_model.h` | GAN, DCGAN, WGAN-GP, 训练循环 |
| `diffusion_model.h` | DDPM, DDIM, Classifier-Free Guidance |

## 构建

```bash
make
```

## 运行示例

```bash
./example_cnn
./example_rnn
./example_transformer
./demo_gan
./demo_diffusion
```

## 依赖

仅需标准 C99 库 (`math.h`, `stdlib.h`, `string.h`, `time.h`)，无外部依赖。

## 许可

MIT
