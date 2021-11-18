# NiN（Network in Network）

NiN（Network in Network）是Min Lin等人在2014的论文《Network in Network》中提出的一种结构，用于增强模型对感受野内局部块的可分辨性。作者以结构更复杂的方式构造微型神经网络（多层感知机）来抽象感受野内的数据。通过堆叠这样的微型神经网络可以构造更深的NiN。此外，作者在分类层的特征图上使用了全局平均池化（global average pooling）增强模型局部建模能力，这比传统的全连接层更不容易过拟合（参数少）。  
## NiN结构
### mlpconv layer
作者认为CNN当中的卷积滤波器是一种通用线性模型（generalized linear model ，即GLM），抽象的能力较低。所谓抽象指的是**特征对于相同概念的变体是不变的**。使用更强大的非线性函数逼近器可以增强局部模型的抽象能力。
在NiN中，GLM被一种“微型网络”结构替代，这种微型网络是一种通用非线性函数逼近器。这种微型神经网络也被称为mplconv layer，其与线性卷积层的对比如图所示

![](images\layercompare.png)
> 图（b）中类似全连接的部分其实就是大小为1x1的卷积层。  

NiN的整体结构就是由多个mlpconv layer堆叠而成。  
在CNN当中，更深层的滤波器会映射原始输入更大的区域，通过结合浅层的低级概念可以产生高级概念。因此作者认为，在浅层低级概念结合成高级概念之前，对其局部块进行更好的抽象是有益的。
### global average pooling
CNN当中通常用全连接层进行分类。而作者直接用最后一层mlpconv layer的输出特征图，通过一层global average pooling layer进行平均后再进行softmax，作为输出类别的置信度。作者认为，全连接层容易导致过拟合，而且比较依赖dropout正则化，而global average pooling本身就是一种正则化。

### 整体结构
最初的NiN结构是在AlexNet后不久提出的，结构和训练方式与AlexNet相似。  
论文结构如图所示：
![](images\paper.png)
- 激活函数采用relu
- mlpconv layer后面是max pooling layer。作者对比了maxout layer（也就是max pooling）和mlpconv layer，认为mlpconv layer与maxout layer的不同之处在于mlpconv layer用更通用的函数逼近器替代了maxout layer的凸函数逼近器，通用的函数逼近器对潜在概念的不同分布具有更强大的建模能力
- 除了最后一层mlpconv layer，其它的mlpconv layer的输出都加入了dropout（作者发现在mlpconv layer之间加入dropout可以提高模型泛化能力）

更详细的结构如图所示：
![](images\detail.png)
## 实验
作者用了四种基准数据集测试模型性能，分别是CIFAR-10、CIFAR-100、SVHN和MNIST。测试误差结果如图所示
![](images\cifar10.png)
![](images\cifar100.png)
![](images\svhn.png)
![](images\mnist.png)
作者也在CIFAR-10数据集上比较了使用global average pooling和使用全连接的网络的性能
![](images\mlpconvvsfc.png)
另外作者对最后一层mlpconv layer的可视化也挺有意思的。与输入图片真实类别相对应的特征图可以观察到最大的激活（即红色方框内大块的白色部分），而且最大的激活大致出现在和物体在原图内相同的区域。
![](images\visual.png)

## 总结
作者对分类任务提出了一种新的深层网络NiN。NiN由mlpconv layer组成，其以多层感知机的方式对输入进行卷积。同时用global average pooling替代传统CNN中的全连接层。mlpconv layer对局部块的建模效果更好，而且global average pooling作为一种结构正则化可以有效防止过拟合。

---

# 代码
代码依旧由数据集划分文件`split_data.py`，模型文件`nin.py`，训练文件`train.py`以及预测文件`precit.py`组成。
数据集下载地址：<http://download.tensorflow.org/example_images/flower_photos.tgz>
## 模型
```
import torch
from torch import nn

# 参考AlexNet设计
class NiN(nn.Module):
    def __init__(self, num_labels):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            self.nin_block(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=384, out_channels=num_labels, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.init_weight()

    def forward(self,x):
        return self.net(x)

    def init_weight(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )

    def test_output_shape(self):
        test_img = torch.rand(size=(1, 3, 227, 227), dtype=torch.float32)
        for layer in self.net:
            test_img = layer(test_img)
            print(layer.__class__.__name__, 'output shape: \t', test_img.shape)

# nin = NiN(num_labels=5)
# nin.test_output_shape()

```

## 训练
```
import os
import json
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from nin import NiN

# 参考AlexNet的训练方式
BATCH_SIZE = 64  # 论文128
LR = 0.01  # 论文 0.01
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
EPOCHS = 90  # 论文90

DATASET_PATH = 'data'
MODEL = 'NiN.pth'


def train_device(device='cpu'):
    # 只考虑单卡训练
    if device == 'gpu':
        cuda_num = torch.cuda.device_count()
        if cuda_num >= 1:
            print('device:gpu')
            return torch.device(f'cuda:{0}')
    else:
        print('device:cpu')
        return torch.device('cpu')


def dataset_loader(dataset_path):
    dataset_path = os.path.join(os.getcwd(), dataset_path)
    assert os.path.exists(dataset_path), f'[{dataset_path}] does not exist.'
    train_dataset_path = os.path.join(dataset_path, 'train')
    val_dataset_path = os.path.join(dataset_path, 'val')
    # 训练集图片随机裁剪224x224区域，以0.5的概率水平翻转
    # 由于torchvision没有封装PCA jitter，所以用Corlor jitter模拟RGB通道强度的变化（不够严谨...）
    # alexnet中训练样本分布为零均值分布，这里采用了常用的均值为0方差为1的标准正态分布
    data_transform = {
        'train': transforms.Compose([transforms.RandomResizedCrop(size=224),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'val': transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    train_dataset = datasets.ImageFolder(root=train_dataset_path, transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=val_dataset_path, transform=data_transform['val'])
    return train_dataset, val_dataset


def idx2class_json(train_dataset):
    class2idx_dic = train_dataset.class_to_idx
    idx2class_dic = dict((val, key) for key, val in class2idx_dic.items())
    # json.dumps()把python对象转换成json格式的字符串
    json_str = json.dumps(idx2class_dic)
    with open('class_idx.json', 'w') as json_file:
        json_file.write(json_str)
    print('write class_idx.json complete.')


def evaluate_val_accuracy(net, val_dataset_loader, val_dataset_num, device=torch.device('cpu')):
    # ==============================================
    # isinstance()与type()区别：
    # type()不会认为子类是一种父类类型，不考虑继承关系。
    # isinstance()会认为子类是一种父类类型，考虑继承关系。
    # 如果要判断两个类型是否相同推荐使用isinstance()
    # ==============================================
    if isinstance(net, nn.Module):
        net.eval()
    val_correct_num = 0
    for i, (val_img, val_label) in enumerate(val_dataset_loader):
        val_img, val_label = val_img.to(device), val_label.to(device)
        output = net(val_img)
        _, idx = torch.max(output.data, dim=1)
        val_correct_num += torch.sum(idx == val_label)
    val_correct_rate = val_correct_num / val_dataset_num
    return val_correct_rate


def train(net, train_dataset, val_dataset, device=torch.device('cpu')):
    train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
    print(f'[{len(train_dataset)}] images for training, [{len(val_dataset)}] images for validation.')
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(params=net.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)  # 论文使用的优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    # 学习率调整策略
    # 论文中，alexnet将错误率（应该指的是验证集）作为指标，当错误率一旦不再下降的时候降低学习率。alexnet训练了大约90个epoch，学习率下降3次
    # 第一种策略，每30个epoch降低一次学习率（不严谨）
    lr_scheduler=optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30,gamma=0.1)
    # 第二种策略，错误率不再下降的时候降低学习率，我们后面会计算验证集的准确率，错误率不再下降和准确率不再提高是一个意思,所以mode为max，但是
    # 实测的时候
    # ==================================================================================================================
    # 注意：在PyTorch 1.1.0之前的版本，学习率的调整应该被放在optimizer更新之前的。如果我们在 1.1.0 及之后的版本仍然将学习率的调整
    # （即 scheduler.step()）放在 optimizer’s update（即 optimizer.step()）之前，那么 learning rate schedule 的第一个值将
    # 会被跳过。所以如果某个代码是在 1.1.0 之前的版本下开发，但现在移植到 1.1.0及之后的版本运行，发现效果变差，
    # 需要检查一下是否将scheduler.step()放在了optimizer.step()之前。
    # ==================================================================================================================
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=3,
    #                                                     min_lr=0.00001)
    # 在训练的过程中会根据验证集的最佳准确率保存模型
    best_val_correct_rate = 0.0
    for epoch in range(EPOCHS):
        net.train()
        # 可视化训练进度条
        train_bar = tqdm(train_dataset_loader)
        # 计算每个epoch的loss总和
        loss_sum = 0.0
        for i, (train_img, train_label) in enumerate(train_bar):
            optimizer.zero_grad()
            train_img, train_label = train_img.to(device), train_label.to(device)
            output = net(train_img)
            loss = loss_function(output, train_label)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            train_bar.desc = f'train epoch:[{epoch + 1}/{EPOCHS}], loss:{loss:.5f}'
        # 测试验证集准确率
        val_correct_rate = evaluate_val_accuracy(net, val_dataset_loader, len(val_dataset), device)
        # 根据验证集准确率更新学习率
        # lr_scheduler.step(val_correct_rate)
        lr_scheduler.step()
        print(
            f'epoch:{epoch + 1}, '
            f'train loss:{(loss_sum / len(train_dataset_loader)):.5f}, '
            f'val correct rate:{val_correct_rate:.5f}')
        if val_correct_rate > best_val_correct_rate:
            best_val_correct_rate = val_correct_rate
            # 保存模型
            torch.save(net.state_dict(), MODEL)
    print('train finished.')


if __name__ == '__main__':
    # 这里数据集只有5类
    nin = NiN(num_labels=5)
    device = train_device('gpu')
    train_dataset, val_dataset = dataset_loader(DATASET_PATH)
    # 保存类别对应索引的json文件，预测用
    idx2class_json(train_dataset)
    train(nin, train_dataset, val_dataset, device)

```

## 预测
```
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from nin import NiN

# daisy dandelion rose sunflower tulip

IMG_PATH = 'test_img/tulip.jpg'
JSON_PATH = 'class_idx.json'
WEIGHT_PATH = 'NiN.pth'


def predict(net, img, json_label):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    original_img=img
    img = data_transform(img)  # 3,224,224
    img = torch.unsqueeze(img, dim=0)  # 1,3,224,224
    assert os.path.exists(WEIGHT_PATH), f'file {WEIGHT_PATH} does not exist.'
    net.load_state_dict(torch.load(WEIGHT_PATH))
    net.eval()
    with torch.no_grad():
        output = torch.squeeze(net(img))  # net(img)的size为1,5，经过squeeze后变为5
        predict = torch.softmax(output, dim=0)
        predict_label_idx=int(torch.argmax(predict))
        predict_label=json_label[str(predict_label_idx)]
        predict_probability=predict[predict_label_idx]
    predict_result=f'class:{predict_label}, probability:{predict_probability:.3f}'
    plt.imshow(original_img)
    plt.title(predict_result)
    print(predict_result)
    plt.show()


def read_json(json_path):
    assert os.path.exists(json_path), f'{json_path} does not exist.'
    with open(json_path, 'r') as json_file:
        idx2class = json.load(json_file)
        return idx2class


if __name__ == '__main__':
    net = NiN(num_labels=5)
    img = Image.open(IMG_PATH)
    idx2class = read_json(JSON_PATH)
    predict(net, img, idx2class)

```
### 预测结果
![](\predict_result\daisy.png)
![](\predict_result\dandelion.png)
![](\predict_result\rose.png)
![](\predict_result\sunflower.png)
![](\predict_result\tulip.png)
rose和tulip预测错误，其余正确。

## 总结
训练10个epoch的时候还是发现验证集准确率一直在24%。调整batchsize=64，优化器为adam，初始化学习率为0.0002，训练90个epoch，每30个epoch学习率除以10。最终验证集准确率达到63%。从生成模型文件的大小来看参数确实降低了许多（AlexNet大约220MB，NiN只需要78MB）。

--- 

参考：

- [NIN网络-Network In Network]<https://blog.csdn.net/fanzy1234/article/details/86173123>
- [(翻译)Network In Network]<https://www.jianshu.com/p/8a3f9f06bbe3>
- [从头学pytorch(十七):网络中的网络NIN]<https://www.cnblogs.com/sdu20112013/p/12181314.html>
- [神经网络系列（四）--NIN网络结构]<https://blog.csdn.net/budong282712018/article/details/102787380>
- [（NIN网络）Network in Network论文阅读笔记]<https://zhuanlan.zhihu.com/p/138829008>
- [NiN：使用1×1卷积层替代全连接层]<https://zhuanlan.zhihu.com/p/337045965>