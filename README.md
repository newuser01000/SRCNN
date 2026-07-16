# SRCNN
使用经典卷积神经网络用于遥感影像超分，采用Sentinel-2遥感影像。由于无对应20m近红外波段作为模型输入，模型分为可见光和近红外两个模型进行训练，超分结果按照B8、B4、B3、B2波段顺序保存。  
数据格式为GeoTIFF
# 目录
<a href="#train">train</a>  
<a href="#test">test</a>  
<a href="#demo">demo</a>  
<a href="#references">references</a>  

# train
开始训练
``` python
  python train.py --steps=10000         \
                  --architecture="915"  \
                  --batch_size=128      \
                  --save-best-only=1    \
                  --save-every=1000     \
                  --save-log=1          \
                  --ckt-dir="checkpoint/SRCNN915/x2/AdamW/B234"  \
                  --scale=2             \
                  --mode="B234"
```
- --save-best-only:如果取值为1，模型将只保留最佳模型，否则将保存每一save-every步。
- --save-log:如果取值为1，训练损失、训练指标、验证损失、验证指标都将在每save-every步被保存。
- --scale:放大倍率，需要自己构建数据集
- --mode：训练的模型，B324/B8
- --ckt-dir：checkpoint/SRCNN{architecture}/x{scale}/AdamW/B{mode}  
注意：如果要训练一个新模型，需要删除--ckt-dir目录下所有文件。模型会在训练时检查检查点文件，若存在，则会从最新检查点处继续训练。
# test
自己构建测试集。训练结束后，可以测试模型的训练效果，结果是计算所有图像的平均PSNR。  
``` python
  python test.py --scale=2             \
                 --architecture="915"  \
                 --ckp-path="default"  \
                 --mode="B8"
```
# demo
训练结束后，可以用该命令测试模型，结果是GeoTIFF文件。  
``` python
  python demo.py --scale=2             \
                 --architecture="915"  \
                 --image-path=""       \
                 --out_sr=""           \
                 --out_bicubic=""      \
                 --b234-ckpt-path=""   \
                 --b8-ckpt-path=""     
```
# references
SRCNN code： https://github.com/Nhat-Thanh/SRCNN-TF
