"""Training loop."""

import sys

import torch
from torch.utils.data import DataLoader

from gcn_prot.features import batched_eucl, transform_input
from gcn_prot.visualization import plot_epoch

from .utils import calc_accuracy


def forward_step(batch, model, training):
    """Pass forward.

    Paramters
    ---------
    batch: tuple(torch.Tensor)
        from DataLoader
    model: torch.nn.Module
    training: bool
        is network training
    cuda: bool

    """
    inputs, labels_onehot = transform_input(batch, training)
    v, c = inputs
    if model.in_cuda:
        v, c = v.cuda(), c.cuda()
        labels_onehot = labels_onehot.cuda()
    inputs = v, batched_eucl(c)
    predictions = model(inputs)

    return predictions, labels_onehot


def run_epoch(
    model, iterator, optimizer, criterion, debug=False, epoch=None, training=True,
):
    """Train an epoch."""
    epoch_loss = 0
    acc = 0
    n = len(iterator)
    for i, batch in enumerate(iterator):
        if debug:
            sys.stdout.write(f"\rEpoch {epoch} ({i}/{n})            ")
            sys.stdout.flush()
        predictions, labels = forward_step(batch, model, training)

        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        acc += calc_accuracy(predictions, labels)
    return epoch_loss / n, acc / n


def fit_network(
    model,
    train_dataset,
    test_dataset,
    optimizer,
    criterion,
    batch_size,
    epochs=100,
    plot_every=1,
    debug=False,
    save=False,
):
    """Run epochs of training and testing on a NN `model`.

    Parameters
    ----------
    model: torch.nn.Module
    iterator: torch.utils.data.Dataset
    optimizer: torch.optim
    criterion: torch.nn.modules.loss
    batch_size: int
    epochs: int
    plot_every: int
    frequency in epochs when the network will be plotted
    cuda: bool, default False
        use GPU
    debug: bool, default False
        will print a progress bar for each epoch
    save: str
        if a string is supplied the model will be saved everytime it surpasses
        the loss (on the test set) of any other model

    Returns
    -------
    model: torch.nn.Module
        trained model

    """
    trainloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, drop_last=False
    )
    testloader = DataLoader(
        test_dataset, shuffle=True, batch_size=batch_size, drop_last=False,
    )
    all_train = []
    all_test = []
    all_epochs = []
    acc_train = []
    acc_test = []
    best_test_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        tr_loss, tr_acc = run_epoch(
            model, trainloader, optimizer, criterion, debug, epoch, training=True
        )
        model.eval()
        te_loss, te_acc = run_epoch(
            model, testloader, optimizer, criterion, debug, epoch, training=False,
        )

        all_train.append(tr_loss)
        acc_train.append(tr_acc)
        all_test.append(te_loss)
        acc_test.append(te_acc)
        all_epochs.append(epoch)

        if te_loss < best_test_loss and save:
            best_test_loss = te_loss
            torch.save(model.state_dict(), save)

        if epoch % plot_every == 0 and epoch != 0:
            plot_epoch(all_epochs, all_train, all_test, acc_train, acc_test, epoch)

    return model
