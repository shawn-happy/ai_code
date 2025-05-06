# 使用L1非结构化剪枝方法对LeNet进行全局剪枝操作，要求剪掉30%的参数，并输出每一层被剪枝的比例
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.reshape(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


'全局剪枝'
if __name__ == '__main__':
    """
    一种常见的、可能更强大的技术,是通过删除(例如)整个模型中最低30%的连接，
    而不是删除每个层中最低30%的连接，一次性删除整个模型。这可能导致每层的修剪百分数不同。
    """
    model = LeNet()
    parameters_to_prune = (
        (model.conv1, 'weight'),
        (model.conv2, 'weight'),
        (model.fc1, 'weight'),
        (model.fc2, 'weight'),
        (model.fc3, 'weight'),
    )
    #剪枝方法、比例
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=0.3
    )

#统计每层权重被剪枝的百分比
    print(
        "Sparsity in conv1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv1.weight == 0))
            / float(model.conv1.weight.nelement())
        )
    )
    print(
        "Sparsity in conv2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.conv2.weight == 0))
            / float(model.conv2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc1.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc1.weight == 0))
            / float(model.fc1.weight.nelement())
        )
    )
    print(
        "Sparsity in fc2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc2.weight == 0))
            / float(model.fc2.weight.nelement())
        )
    )
    print(
        "Sparsity in fc3.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc3.weight == 0))
            / float(model.fc3.weight.nelement())
        )
    )
    # 统计全部权重被剪枝的百分比
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.conv1.weight == 0)
                + torch.sum(model.conv2.weight == 0)
                + torch.sum(model.fc1.weight == 0)
                + torch.sum(model.fc2.weight == 0)
                + torch.sum(model.fc3.weight == 0)
            )
            / float(
                model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
                + model.fc3.weight.nelement()
            )
        )
    )

if __name__ == '__main__':
    model = LeNet()
