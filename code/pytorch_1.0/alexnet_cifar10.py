import torch
from torchvision import datasets, models, transforms


def main():
    # 載入dataset : CIFAR10 (torchvision包裝好的)
    transform = None
    cifar10_datasets = {
        x: datasets.CIFAR10(
            root='../data/CIFA10',
            train=True,
            transform=transform,
            download=True)
        for x in ['train', 'val']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            cifar10_datasets[x], batch_size=4, shuffle=True, num_workers=2)
        for x in ['train, val']
    }
    dataset_sizes = {x: len(cifar10_datasets) for x in ['train', 'val']}

    # 設定device參數
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()