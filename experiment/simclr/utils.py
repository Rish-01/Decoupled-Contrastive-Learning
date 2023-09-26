import torch
from PIL import Image
from torchvision import transforms

def create_pair_dataset(parent_dataset):
    class paired_dataset(parent_dataset):

        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            # img = img.numpy()
            img = Image.fromarray(img)

            if self.transform is not None:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return pos_1, pos_2, target
    return paired_dataset


mnist_train_transform = transforms.Compose([
    transforms.Resize((32, 32), antialias=True),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.squeeze(x)),
    transforms.Lambda(lambda x: torch.stack([x, x, x], 0)),
    transforms.RandomResizedCrop(32, antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

mnist_test_transform = transforms.Compose([
    transforms.Resize((32, 32), antialias=True),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.squeeze(x)),
    transforms.Lambda(lambda x: torch.stack([x, x, x], 0)),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32, antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
