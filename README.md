
## 前言


本文介绍使用神经网络进行实战。
使用的代码是《零基础学习人工智能—Python—Pytorch学习（九）》里的代码。


## 代码实现


#### mudule定义


首先我们自定义一个module，创建一个torch\_test17\_Model.py文件(这个module要单独用个py文件定义)，如下：



```
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        return x

```

#### module创建


编写创建module的py文件，代码如下：



```
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch_test17_Model as tm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784 
hidden_size = 100
num_classes = 10
batch_size = 100
learning_rate = 0.001
num_epochs = 200 # 要训练200-400轮效果最好


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

train_loader = torch.utils. data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True) 

model = tm.ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
print("number total epochs(训练的回合):",num_epochs)
print("number total steps(训练的次数):",n_total_steps)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images.shape: torch.Size([100, 3, 32, 32]) 
        # images张量的四个维度是(B, C, H, W)
        # B 是批量大小（即图像的数量）。
        # C 是图像的通道数（例如，RGB 图像的通道数是 3）。
        # H 和 W 分别是图像的高度和宽度。
        print("images.shape:", images.shape) #100行，后面的维度是3,32,32。这个是图片信息。
        # lables是对应images这100个图片的标签
        print("labels.shape:", labels.shape)
        print("labels[0].item():", labels[0].item())  # 输出例子 labels[0].item()=6
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        print("loss.item()",loss.item())  # 输出例子 loss.item()=2.300053596496582
        # 逆向传播和优化
        optimizer.zero_grad()
        loss.backward() ##执行逆向传播 会使用criterion的函数关系求偏导，然后把x的值，带入偏导公式求值，然后再乘以loss，得到新x值
        optimizer.step()
    print(f'训练轮次Epoch [{epoch}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    print('==================')
print('训练结束')


filePath = "model.pth" #没有路径，会保存到python文件所在目录
torch.save(model, filePath)
print('保存完成')

```

代码会输出loss的值，我们要重点关注这个值。
Loss 值越大，表示模型的预测与真实标签之间的差距较大，模型的性能较差。
Loss 值越小，表示模型的预测更接近真实标签，性能逐渐提高。
即，loss值接近0的时候，这个模型就可以用了。


#### module使用


编写使用module验证图片的py文件，注意要引用torch\_test17\_Model.py文件，代码如下：



```
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch_test17_Model as tm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100

transform = transforms.Compose(
    [transforms.Resize((32, 32)),# 如果预测时处理的图片尺寸与训练时不同，如评估输入的图片尺寸为 [100, 3, 64, 64]，而模型训练使用的尺寸是 [100, 3, 32, 32],可以用這個转换一下
     transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)


filePath = "model.pth" #没有路径，会保存到python文件所在目录
model = torch.load(filePath,weights_only=False)
model.eval() # 切换到评估模式


############################使用阈值判断######################################
threshold = 0.7  # 设定一个阈值，表示模型的信心度，用阈值判断的话，要求模型必须更精确，如果只是两轮的训练，会出现全部判定不过去的情况
with torch.no_grad():
    for images, labels in test_loader:
        print("############################判断######################################")
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        print("outputs.shape",outputs.shape)
        # 计算 softmax 概率
        probabilities = F.softmax(outputs, dim=1)

        max_probs, predicted = torch.max(probabilities, 1)
        for i in range(len(predicted)):
            if max_probs[i] < threshold:  # 如果置信度低于阈值，认为是未知类别
                print(f"图片 {i} 被认为是未知类别，置信度 {max_probs[i]:.4f}")
            else:
                print(f"图片 {i} 被认为是类别 {predicted[i]}，置信度 {max_probs[i]:.4f}")

```

判断图片是什么的时候，使用阈值模式。


## 结语


到此，我们对于神经网络，卷积神经网络，深度网络都有了一定了解。
然后我们就可以继续学习transformer了。




---


传送门：
[零基础学习人工智能—Python—Pytorch学习—全集](https://github.com "零基础学习人工智能—Python—Pytorch学习—全集")




---


注：此文章为原创，任何形式的转载都请联系作者获得授权并注明出处！




---


![](https://img2024.cnblogs.com/blog/243596/202402/243596-20240222170657054-811388484.png)




---


若您觉得这篇文章还不错，请点击下方的【推荐】，非常感谢！


 [https://github.com/kiba/p/18609581](https://github.com):[CMESPEED\-楚门加速器](https://cmnspeed.com)


