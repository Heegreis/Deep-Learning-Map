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


if __name__ == "__main__":
    main()