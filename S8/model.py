import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class S6_Model(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.Dropout(0.0))
        self.conv2 = nn.Sequential(nn.Conv2d(8, 8, 3), nn.BatchNorm2d(8), nn.ReLU(), nn.Dropout(0.00))
        self.conv3 = nn.Sequential(nn.Conv2d(8, 16, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout(0.00))
        self.conv4 = nn.Sequential(nn.Conv2d(16, 16, 3), nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout(0.00))
        self.conv5 = nn.Sequential(nn.Conv2d(16, 16, 3), nn.BatchNorm2d(16), nn.ReLU(), nn.Dropout(0.01))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv6 = nn.Sequential(nn.Conv2d(16, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0.02))
        self.conv7 = nn.Sequential(nn.Conv2d(32, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout(0.00))
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling layer

        self.fc1 = nn.Linear(32, 10)  # Update the input size for the FC layer

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        x= self.pool1(self.conv5(self.conv4(x)))
        # x = self.pool2(self.conv4(self.conv3(x)))
        x = self.conv7(self.conv6(x))
        x = self.global_pool(x)  # Apply global average pooling
        x = x.view(-1, 32)  # Flatten the tensor
        x = self.fc1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class S7_Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1) # 28>28 | 3
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1) # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 28 > 14 | 10
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1) # 14> 14 | 12
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1) #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2) # 14 > 7 | 28
        self.conv5 = nn.Conv2d(32, 64, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(64, 64, 3) # 5 > 3 | 32 | 3*3*64 | 3x3x64x10 | 
        self.conv7 = nn.Conv2d(64, 10, 3) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10) #1x1x10> 10
        return F.log_softmax(x, dim=-1)
		
		
		
class S7_Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0)  # Output: 16 channels, Receptive Field: 3x3, Input: 1x28x28
        self.conv2 = nn.Conv2d(16, 16, 3, padding=0)  # Output: 16 channels, Receptive Field: 5x5, Input: 16x26x26
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 16 channels, Receptive Field: 10x10, Input: 16x24x24
        self.conv3 = nn.Conv2d(16, 32, 3, padding=0)  # Output: 32 channels, Receptive Field: 12x12, Input: 16x12x12
        self.conv4 = nn.Conv2d(32, 32, 3, padding=0)  # Output: 32 channels, Receptive Field: 14x14, Input: 32x10x10
        self.conv5 = nn.Conv2d(32, 8, 3)  # Output: 8 channels, Receptive Field: 16x16, Input: 32x8x8
        self.fc = nn.Linear(288, 10)  # Output: 10 classes
    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x=x.view(-1,288)
        #x = x.view(x.size(0), 8)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

		
class S7_Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=0) # 28>26 | 3
        self.conv2 = nn.Conv2d(10, 10, 3, padding=0) # 26 > 24 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 24 > 12 | 10
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0) # 12> 10 | 12
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0) #10 > 8 | 14
        self.conv5 = nn.Conv2d(16, 8, 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(8, 10, 6) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.conv6(F.relu(x))
        x = x.view(-1, 10) 
        return F.log_softmax(x, dim=-1)		
		

class S7_Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=0) # 28>26 | 3
        self.conv2 = nn.Conv2d(10, 10, 3, padding=0) # 26 > 24 |  5
        self.pool1 = nn.MaxPool2d(2, 2) # 24 > 12 | 10
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0) # 12> 10 | 12
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0) #10 > 8 | 14
        self.conv5 = nn.Conv2d(16, 20 , 3) # 7 > 5 | 30
        self.conv6 = nn.Conv2d(20, 10, 3) # 3 > 1 | 34 | > 1x1x10
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.conv5(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.conv6(F.relu(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)




class S7_Model5(nn.Module):
    def __init__(self):
        super(Model5, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, padding=0) # 28>26 | 3
        self.bn1 = nn.BatchNorm2d(10)
        self.dropout1 = nn.Dropout2d(0)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=0) # 26 > 24 |  5
        self.bn2 = nn.BatchNorm2d(10)
        self.dropout2 = nn.Dropout2d(0)
        self.pool1 = nn.MaxPool2d(2, 2) # 24 > 12 | 10
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0) # 12> 10 | 12
        self.bn3 = nn.BatchNorm2d(10)
        self.dropout3 = nn.Dropout2d(0.02)
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0) #10 > 8 | 14
        self.bn4 = nn.BatchNorm2d(16)
        self.dropout4 = nn.Dropout2d(0.05)
        self.conv5 = nn.Conv2d(16, 16 , 3) # 7 > 5 | 30
        self.bn5 = nn.BatchNorm2d(16)
        self.dropout5 = nn.Dropout2d(0.05)
        self.conv6 = nn.Conv2d(16, 16 , 3)
        self.bn6 = nn.BatchNorm2d(16)
        self.dropout6 = nn.Dropout2d(0.01)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv7 = nn.Conv2d(16, 10, 1) # 3 > 1 | 34 | > 1x1x10

    def forward(self, x):
        x = self.pool1(self.dropout2(self.bn2(F.relu(self.conv2(self.dropout1(self.bn1(F.relu(self.conv1(x)))))))))
        x = self.conv5(self.dropout4(self.bn4(F.relu(self.conv4(self.dropout3(self.bn3(F.relu(self.conv3(x)))))))))
        x = self.gap(self.bn6(self.dropout6(F.relu(self.conv6(self.dropout5(self.bn5(F.relu(x))))))))
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)





class S8_Model_BN(nn.Module):
    def __init__(self, norm='bn'):
        super(Model5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, 32)
        self.dropout1 = nn.Dropout2d(0.02)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(2, 64)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, 64)
        self.dropout2 = nn.Dropout2d(0.02)
        self.conv3 = nn.Conv2d(64, 10, 1, padding=0, bias=False)
        if norm == 'bn':
            self.n3 = nn.BatchNorm2d(10)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(2, 10)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, 10)
        self.dropout3 = nn.Dropout2d(0.02)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(10, 16, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n4 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(2, 16)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1, 16)
        self.dropout4 = nn.Dropout2d(0.02)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n5 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1, 32)
        self.dropout5 = nn.Dropout2d(0.02)
        self.conv6 = nn.Conv2d(32, 64, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n6 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(2, 64)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1, 64)
        self.dropout6 = nn.Dropout2d(0.02)
        self.conv7 = nn.Conv2d(64, 10, 1, padding=0, bias=False)
        if norm == 'bn':
            self.n7 = nn.BatchNorm2d(10)
        elif norm == 'gn':
            self.n7 = nn.GroupNorm(2, 10)
        elif norm == 'ln':
            self.n7 = nn.GroupNorm(1, 10)
        self.dropout7 = nn.Dropout2d(0.02)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(10, 32, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n8 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n8 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n8 = nn.GroupNorm(1, 32)
        self.dropout8 = nn.Dropout2d(0.02)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n9 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n9 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n9 = nn.GroupNorm(1, 32)
        self.dropout9 = nn.Dropout2d(0.02)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv10 = nn.Conv2d(32, 10, 1, padding=0, bias=False)
        if norm == 'bn':
            self.n10 = nn.BatchNorm2d(10)
        elif norm == 'gn':
            self.n10 = nn.GroupNorm(2, 10)
        elif norm == 'ln':
            self.n10 = nn.GroupNorm(1, 10)
        self.dropout10 = nn.Dropout2d(0)


    def forward(self, x):
        x = self.pool1(self.dropout3(self.n3(F.relu(self.conv3(self.dropout2(self.n2(F.relu(self.conv2(\
             self.dropout1(self.n1(F.relu(self.conv1(x)))))))))))))
        x = self.pool2(self.dropout7(self.n7(F.relu(self.conv7(self.dropout6(self.n6(F.relu(self.conv6(self.dropout5(self.n5(F.relu(self.conv5(\
             self.dropout4(self.n4(F.relu(self.conv4(x)))))))))))))))))
        x = self.conv10(self.gap(self.dropout9(self.n9(F.relu(self.conv9(\
             self.dropout8(self.n8(F.relu(self.conv8(x))))))))))
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)





class S8_Model_GN(nn.Module):
    def __init__(self, norm='gn'):
        super(Model5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, 32)
        self.dropout1 = nn.Dropout2d(0.02)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(2, 64)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, 64)
        self.dropout2 = nn.Dropout2d(0.02)
        self.conv3 = nn.Conv2d(64, 10, 1, padding=0, bias=False)
        if norm == 'bn':
            self.n3 = nn.BatchNorm2d(10)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(2, 10)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, 10)
        self.dropout3 = nn.Dropout2d(0.02)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(10, 16, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n4 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(2, 16)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1, 16)
        self.dropout4 = nn.Dropout2d(0.02)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n5 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1, 32)
        self.dropout5 = nn.Dropout2d(0.02)
        self.conv6 = nn.Conv2d(32, 64, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n6 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(2, 64)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1, 64)
        self.dropout6 = nn.Dropout2d(0.02)
        self.conv7 = nn.Conv2d(64, 10, 1, padding=0, bias=False)
        if norm == 'bn':
            self.n7 = nn.BatchNorm2d(10)
        elif norm == 'gn':
            self.n7 = nn.GroupNorm(2, 10)
        elif norm == 'ln':
            self.n7 = nn.GroupNorm(1, 10)
        self.dropout7 = nn.Dropout2d(0.02)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(10, 32, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n8 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n8 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n8 = nn.GroupNorm(1, 32)
        self.dropout8 = nn.Dropout2d(0.02)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n9 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n9 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n9 = nn.GroupNorm(1, 32)
        self.dropout9 = nn.Dropout2d(0.02)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv10 = nn.Conv2d(32, 10, 1, padding=0, bias=False)
        if norm == 'bn':
            self.n10 = nn.BatchNorm2d(10)
        elif norm == 'gn':
            self.n10 = nn.GroupNorm(2, 10)
        elif norm == 'ln':
            self.n10 = nn.GroupNorm(1, 10)
        self.dropout10 = nn.Dropout2d(0)


    def forward(self, x):
        x = self.pool1(self.dropout3(self.n3(F.relu(self.conv3(self.dropout2(self.n2(F.relu(self.conv2(\
             self.dropout1(self.n1(F.relu(self.conv1(x)))))))))))))
        x = self.pool2(self.dropout7(self.n7(F.relu(self.conv7(self.dropout6(self.n6(F.relu(self.conv6(self.dropout5(self.n5(F.relu(self.conv5(\
             self.dropout4(self.n4(F.relu(self.conv4(x)))))))))))))))))
        x = self.conv10(self.gap(self.dropout9(self.n9(F.relu(self.conv9(\
             self.dropout8(self.n8(F.relu(self.conv8(x))))))))))
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)


class S8_Model_LN(nn.Module):
    def __init__(self, norm='ln'):
        super(Model5, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, 32)
        self.dropout1 = nn.Dropout2d(0.02)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(2, 64)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, 64)
        self.dropout2 = nn.Dropout2d(0.02)
        self.conv3 = nn.Conv2d(64, 10, 1, padding=0, bias=False)
        if norm == 'bn':
            self.n3 = nn.BatchNorm2d(10)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(2, 10)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, 10)
        self.dropout3 = nn.Dropout2d(0.02)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(10, 16, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n4 = nn.BatchNorm2d(16)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(2, 16)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1, 16)
        self.dropout4 = nn.Dropout2d(0.02)
        self.conv5 = nn.Conv2d(16, 32, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n5 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1, 32)
        self.dropout5 = nn.Dropout2d(0.02)
        self.conv6 = nn.Conv2d(32, 64, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n6 = nn.BatchNorm2d(64)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(2, 64)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1, 64)
        self.dropout6 = nn.Dropout2d(0.02)
        self.conv7 = nn.Conv2d(64, 10, 1, padding=0, bias=False)
        if norm == 'bn':
            self.n7 = nn.BatchNorm2d(10)
        elif norm == 'gn':
            self.n7 = nn.GroupNorm(2, 10)
        elif norm == 'ln':
            self.n7 = nn.GroupNorm(1, 10)
        self.dropout7 = nn.Dropout2d(0.02)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv8 = nn.Conv2d(10, 32, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n8 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n8 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n8 = nn.GroupNorm(1, 32)
        self.dropout8 = nn.Dropout2d(0.02)
        self.conv9 = nn.Conv2d(32, 32, 3, padding=0, bias=False)
        if norm == 'bn':
            self.n9 = nn.BatchNorm2d(32)
        elif norm == 'gn':
            self.n9 = nn.GroupNorm(2, 32)
        elif norm == 'ln':
            self.n9 = nn.GroupNorm(1, 32)
        self.dropout9 = nn.Dropout2d(0.02)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv10 = nn.Conv2d(32, 10, 1, padding=0, bias=False)
        if norm == 'bn':
            self.n10 = nn.BatchNorm2d(10)
        elif norm == 'gn':
            self.n10 = nn.GroupNorm(2, 10)
        elif norm == 'ln':
            self.n10 = nn.GroupNorm(1, 10)
        self.dropout10 = nn.Dropout2d(0)


    def forward(self, x):
        x = self.pool1(self.dropout3(self.n3(F.relu(self.conv3(self.dropout2(self.n2(F.relu(self.conv2(\
             self.dropout1(self.n1(F.relu(self.conv1(x)))))))))))))
        x = self.pool2(self.dropout7(self.n7(F.relu(self.conv7(self.dropout6(self.n6(F.relu(self.conv6(self.dropout5(self.n5(F.relu(self.conv5(\
             self.dropout4(self.n4(F.relu(self.conv4(x)))))))))))))))))
        x = self.conv10(self.gap(self.dropout9(self.n9(F.relu(self.conv9(\
             self.dropout8(self.n8(F.relu(self.conv8(x))))))))))
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)
