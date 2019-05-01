import torch
import torchvision
import torchvision.transforms as transforms


def get_train_data(data_path, batch_size, test_batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = torchvision.datasets.MNIST(root=data_path,
                                           train=True,
                                           download=True,
                                           transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)

    test_set = torchvision.datasets.MNIST(root=data_path,
                                          train=False,
                                          download=True,
                                          transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=2)

    return train_loader, test_loader
