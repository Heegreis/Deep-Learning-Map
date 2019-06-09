import torch
from torchvision import datasets, models, transforms


def test(model, device, test_loader):
    pass


def main():
    # 載入dataset : CIFAR10 (torchvision包裝好的)
    transform = None
    cifar10_train_set = datasets.CIFAR10(root='../data/CIFA10',
                                         train=True,
                                         transform=transform,
                                         download=True)
    cifar10_test_set = datasets.CIFAR10(root='../data/CIFA10',
                                        train=False,
                                        transform=transform,
                                        download=True)
    train_loader = torch.utils.data.DataLoader(cifar10_train_set,
                                               batch_size=4,
                                               shuffle=True,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(cifar10_test_set,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=2)

    # 設定device參數
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #載入模型
    model = models.alexnet(pretrained=True, progress=True)


if __name__ == "__main__":
    main()