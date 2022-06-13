# 图像生成 – Variational AutoEncoders (VAE)

## 代码结构

- data：MNIST数据集
- images：保存了图像生成的结果。其中./images/best_config为最优配置下生成的图像，image_1.png代表了第1个epoch生成的图像；compare_gen.png为VAE对compare_raw.png的重构结果。
- VAE.py：定义了VAE的模型结构
- train.py：VAE的训练及测试代码
- run.sh：运行脚本

## 代码运行

修改run.sh脚本中的STORE_PATH为VAE生成图像结果的保存路径，然后运行

```bash
sbatch run.sh
```

即可复现./images/best_config的结果（但在不同机器上可能会有不同的生成效果）