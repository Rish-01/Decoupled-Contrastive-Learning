from tqdm import tqdm
import torch
from torch import nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import utils
from model import Model
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from torchvision.datasets.cifar import CIFAR10, CIFAR100
from torchvision.datasets.stl10 import STL10
from torchvision.datasets import FashionMNIST, MNIST

class encoder(nn.Module):
    def __init__(self, pretrained_path):
        super(encoder, self).__init__()

        # encoder
        self.f = Model().f
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return feature
    
def select_dataset(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        return CIFAR10
    if dataset_name == 'cifar100':
        return CIFAR100
    if dataset_name == 'stl10':
        return STL10
    if dataset_name == 'fmnist':
        return FashionMNIST
    if dataset_name == 'mnist':
        return MNIST
    raise ValueError("Invalid dataset name")


def save_tsne_plot():

    parser = argparse.ArgumentParser(description='Linear Evaluation')
    parser.add_argument('--model_path', type=str, default='results/checkpoints/128_0.5_200_512_500_model.pth',
                        help='The pretrained model path')
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name: cifar10, cifar100, stl10')
    parser.add_argument('--loss', default='ce', type=str, help='loss function')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = encoder(f'{args.model_path}')
    model = model.to(device)

    dataset_class = select_dataset(args.dataset)

    if args.dataset == 'MNIST' or args.dataset == 'FMNIST':
        train_data = dataset_class(root='../../data', train=True, transform=utils.mnist_train_transform, download=True)
        test_data = dataset_class(root='../../data', train=False, transform=utils.mnist_test_transform, download=True)
      
        concat_dataset = ConcatDataset([train_data, test_data])
        train_test_loader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)


    else:
        train_data = dataset_class(root='../../data', train=True, transform=utils.train_transform, download=True)
        test_data = dataset_class(root='../../data', train=False, transform=utils.test_transform, download=True)

        concat_dataset = ConcatDataset([train_data, test_data])
        train_test_loader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

    fig, ax = plt.subplots()
    
    embeddings_accumulated = []
    embeddings_tsne_accumulated = []
    for i, (images, labels) in enumerate(tqdm(train_test_loader)):
        model.eval()
        images = images.to(device)
        
        # Run inference by forward propagating through the encoder
        with torch.no_grad():
            encoder_embeddings = model(images)

        # Calculate TSNE embeddings
        tsne = TSNE(n_components=2, random_state=42)
        encoder_embeddings = encoder_embeddings.cpu()
        tsne_embeddings = tsne.fit_transform(encoder_embeddings)

        # Plot the tsne embeddings of the pretrained encoder
        sc = ax.scatter(x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], c=labels, s=0.5)
        ax.legend(*sc.legend_elements(), title='clusters', bbox_to_anchor = (1 , 1))

        # Concat the labels into the embeddings to create an embedding of shape(batch-size, feature-size + 1)
        labels = labels.reshape(-1, 1)
        encoder_embeddings_appended = torch.cat((encoder_embeddings, labels), dim=-1)
        embeddings_accumulated.append(encoder_embeddings_appended)

        # Accumulate tsne embeddings over all batches of shape(batch-size, 2 + 1)
        tsne_embeddings_tensor = torch.tensor(tsne_embeddings)
        tsne_embeddings_tensor = torch.cat((tsne_embeddings_tensor, labels), dim=-1)
        embeddings_tsne_accumulated.append(tsne_embeddings_tensor)
    
    embeddings_accumulated = torch.stack(embeddings_accumulated)
    embeddings_accumulated = embeddings_accumulated.reshape(-1, embeddings_accumulated.shape[-1])
    embeddings_tsne_accumulated = torch.stack(embeddings_tsne_accumulated)
    embeddings_tsne_accumulated = embeddings_tsne_accumulated.reshape(-1, embeddings_tsne_accumulated.shape[-1])
    
    # Save TSNE plot
    fig.savefig(f'../../results/tsne_plots/tsne_{args.dataset}_{args.loss}.png')
    # torch.save(embeddings_accumulated, f'../../results/embeddings/embeddings_{args.dataset}_{args.loss}.pt')
    # torch.save(embeddings_tsne_accumulated, f'../../results/embeddings_tsne/embeddings_tsne_{args.dataset}_{args.loss}.pt')

    # Save embeddings matrices as an npy file
    np.save(f'../../results/embeddings/embeddings_{args.dataset}_{args.loss}.npy', embeddings_accumulated.numpy())
    np.save(f'../../results/embeddings_tsne/embeddings_{args.dataset}_{args.loss}.npy', embeddings_tsne_accumulated.numpy())

    embeddings = np.load('../../results/embeddings_tsne/embeddings_CIFAR10_dclw.npy')
    print(embeddings.shape)


if __name__ == '__main__':
    save_tsne_plot()

