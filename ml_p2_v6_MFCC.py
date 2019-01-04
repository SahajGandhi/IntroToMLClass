import matplotlib

matplotlib.use('Agg')
import torch
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
from python_speech_features import mfcc
from pytorch_nsynth.nsynth import NSynth
import torch.nn as nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model_class import Net2


def dataLoaders(batch_size):
    '''
    This function loads the data for training and testing of the model
    :param batch_size: the size of mini batches
    :return: dataloader objects of train, validation and testing data
    '''
    # audio samples are loaded as an int16 numpy array
    # rescale intensity range as float [-1, 1]
    print("--- Loading data ---")
    toFloat = transforms.Lambda(
        lambda x: (x / np.iinfo(np.int16).max) + 1)  # Added +1 for solving negative number problem
    # normalizeValue = transforms.Normalize(torch.mean(x),torch.std(x))
    # use instrument_family and instrument_source as classification targets
    dataset_Train = NSynth(
        "/local/sandbox/nsynth/nsynth-train",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])

    loader_Train = data.DataLoader(dataset_Train, batch_size=batch_size, shuffle=True)

    dataset_Valid = NSynth(
        "/local/sandbox/nsynth/nsynth-valid",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
    loader_Valid = data.DataLoader(dataset_Valid, batch_size=batch_size, shuffle=True)

    dataset_Test = NSynth(
        "/local/sandbox/nsynth/nsynth-test",
        transform=toFloat,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
    loader_Test = data.DataLoader(dataset_Test, batch_size=batch_size, shuffle=False)
    return loader_Train, loader_Valid, loader_Test


def train_and_validation(num_epochs, num_samples, train_loader, validation_loader, optimizer, model, criterion, code_counter):
    '''
    This function trains and validates the model
    :param num_epochs: number of epochs
    :param num_samples: number of samples
    :param train_loader: object containing data for training
    :param validation_loader: object containing data for validation
    :param optimizer: the optimizer object
    :param model: the model being trained and validated
    :param criterion: the loss function object
    :param code_counter: the iteration of code being run
    :return: training and validation losses, list of epoch numbers, other data for plotting error curve
    '''
    arr1 = []
    arr2 = []
    arr3 = []
    patience = 25
    best_acc = 0.
    minLoss = 10
    maxLoss = -1
    print("--- Start Training and Validation ---")
    for epoch in range(num_epochs):
        start_time = time.time()
        print(epoch)
        if patience <= 0:
            break
        arr1.append(epoch)
        #################################################Training
        count = 0
        loss_t = 0.0
        for samples, instrument_family_target, instrument_source_target, targets in train_loader:

            samples = samples[:, [range(1, num_samples + 1)]]
            if use_cuda:
                samples, instrument_family_target = samples.cuda(), instrument_family_target.cuda()
                samples, instrument_family_target = Variable(samples), Variable(instrument_family_target)
            temp = []
            for s in samples:
                temp.append(mfcc(s))
            samples = np.array(temp)
            samples = torch.tensor(samples).cuda()

            optimizer.zero_grad()
            out = model(samples.float())

            loss = criterion(out, instrument_family_target)
            loss.backward()
            optimizer.step()
            loss_t += loss.item()
            count += len(instrument_family_target)
            prediction = np.argmax(out.detach().cpu().numpy(), 1)

            true_positives = instrument_family_target.cpu().numpy() == prediction

            running_acc += np.sum(true_positives)

        loss_t = loss_t / count
        arr2.append(loss_t)
        print("Training loss: " + str(loss_t))

        ########################## For plotting convergence line in the plot
        if loss_t < minLoss:
            minLoss = loss_t
        if loss_t > maxLoss:
            maxLoss = loss_t

        ##################################################Validation
        loss_v = 0.0
        count = 0
        running_loss = 0.0
        running_acc = 0.0

        for samples, instrument_family_target, instrument_source_target, targets in validation_loader:

            if use_cuda:
                samples, instrument_family_target = samples.cuda(), instrument_family_target.cuda()
                samples = samples[:, [range(1, num_samples + 1)]]
            temp = []
            for s in samples:
                temp.append(mfcc(s))
            samples = np.array(temp)
            samples = torch.tensor(samples).cuda()

            outputs = model(samples.float())
            loss = criterion(outputs, instrument_family_target)
            running_loss += loss.item()
            prediction = np.argmax(outputs.detach().cpu().numpy(), 1)

            true_positives = instrument_family_target.cpu().numpy() == prediction

            running_acc += np.sum(true_positives)
            loss_v += loss.item()
            count += len(instrument_family_target)

        loss_v = loss_v / count
        print("Validation loss: " + str(loss_v))
        arr3.append(loss_v)

        acc = running_acc / count
        print("Accuracy = " + str(acc))
        if acc > best_acc:
            best_acc = acc
            patience = 25
            bestEpoch = epoch
            torch.save(model, "ml_b_p2_v" + str(code_counter) + ".pth")

        else:
            patience -= 1

        print("Epoch " + str(epoch) + " Time = " + str(time.time() - start_time))
    print("Training and Validation Complete !")
    return arr1, arr2, arr3, bestEpoch, minLoss, maxLoss


def plot_error_curve(arr1, arr2, arr3, bestEpoch, minLoss, maxLoss, code_counter):
    '''
    This function plots the train and validation loss/error curves
    :param arr1: list of epoch numbers
    :param arr2: list of training losses
    :param arr3: list of validation losses
    :param bestEpoch: epoch after which overfitting happens
    :param minLoss: start of axis
    :param maxLoss: end of axis
    :param code_counter: the iteration of code being run
    :return: 0
    '''
    fig, ax = plt.subplots()
    ax.plot(arr1, arr2, label='Training Loss')
    ax.plot(arr1, arr3, label='Validation Loss')
    ax.plot([bestEpoch, bestEpoch], [minLoss, maxLoss], color='k', linestyle='-', linewidth=2, label='Overfitting Point')
    ax.set_title("Average Model Loss over Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Loss")
    ax.legend()

    # Adjust x-axis ticks
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    fig.savefig('./loss_curve_ml_b_p2_v' + str(code_counter))
    plt.clf()
    return 0


def test_data(code_counter, test_loader, num_samples, criterion):
    '''
    This function tests the model
    :param code_counter: the iteration of code being run
    :param test_loader: object containing data for testing
    :param num_samples: number of samples
    :param criterion:  the loss function object
    :return: stats needed for displaying or creating any bar plots/histograms
    '''
    map = {}
    model = torch.load("ml_b_p2_v" + str(code_counter) + ".pth")
    running_loss = 0.0
    running_acc = 0.0
    per_class_running_acc = np.zeros(10)
    count = 0.
    per_class_count = np.zeros(10)
    confusionmatrix = np.zeros((10, 10))
    for samples, instrument_family_target, instrument_source_target, targets in test_loader:
        samples, instrument_family_target = samples.to(device), instrument_family_target.to(device)
        samples = samples[:, [range(num_samples)]]
        temp = []
        for s in samples:
            temp.append(mfcc(s))
        samples = np.array(temp)
        samples = torch.tensor(samples).cuda()

        outputs = model(samples.float())
        loss = criterion(outputs, instrument_family_target)
        running_loss += loss.item()
        valuess = outputs.detach().cpu().numpy()
        prediction = np.argmax(outputs.detach().cpu().numpy(), 1)
        realVal = instrument_family_target.detach().cpu().numpy()
        for i in range(1, len(realVal)):
            confusionmatrix[realVal[i], prediction[i]] += 1
        true_positives = instrument_family_target.cpu().numpy() == prediction
        running_acc += np.sum(true_positives)
        count += len(instrument_family_target)

        for i in range(len(per_class_running_acc)):
            idx = np.where(instrument_family_target.cpu().numpy() == i)[0]
            per_class_tp = prediction[idx] == i
            per_class_running_acc[i] += np.sum(per_class_tp)
            per_class_count[i] += len(idx)

    return confusionmatrix, running_acc, count, running_loss, per_class_running_acc, per_class_count, map, valuess


def print_test_stats_and_save_plots(confusionmatrix, running_acc, count, running_loss, per_class_running_acc, per_class_count, code_counter, map):
    '''
    This function prints and creates accuracy histograms
    :param confusionmatrix: confusion matrix
    :param running_acc: accuracy of the batch
    :param count: total count of samples in the batch
    :param running_loss: loss for the batch
    :param per_class_running_acc: accuracy of each of the classes
    :param per_class_count: count of samples in each class
    :param code_counter: the iteration of code being run
    :return: 0
    '''
    print("Confusion Matrix")
    print(confusionmatrix)
    acc = running_acc / count
    print('test loss: %.3f acc: %.3f' % (running_loss / 10, acc))
    map["total"] = acc
    for i, (a, c) in enumerate(zip(per_class_running_acc, per_class_count)):
        per_class_acc = a / c
        print(i, ":", "%.3f" % per_class_acc)
        map[str(i)] = per_class_acc
    print('Finished Training')
    x_label_list = ['Total', 'Bass', 'Brass', 'Flute', 'Guitar', 'Keyboard', 'Mallet', 'Organ', 'Reed', 'String',
                    'Vocal']
    fig, ax = plt.subplots()
    ax.bar(x_label_list, map.values(), align='center')  # A bar chart
    ax.set_title("Histogram Plot of Classwise-Accuracies and Total Accuracy")
    ax.set_xlabel("Instrument")
    ax.set_ylabel("Accuracy")
    plt.xticks(rotation=15)
    fig.savefig('./bar_chart_' + "ml_b_p2_v" + str(code_counter))
    print("Accuracy Plot Saved !")
    return 0


def main():
    code_counter = 16
    input_size = 13
    hidden_size = 80
    num_layers = 5
    num_classes = 10
    batch_size = 100
    num_epochs = 50
    learning_rate = 0.025
    momentum = 0.9
    num_samples = 16000
    train_loader, validation_loader, test_loader = dataLoaders(batch_size)
    print("Data Loaded !")

    model = Net2(input_size, hidden_size, num_layers, num_classes).to(device)
    print("Model Loaded !")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    options = input("Enter letter a for Train-Validate-Test and b for just Test")

    if options.lower() == 'a':
        #################    TRAINING    ##################
        arr1, arr2, arr3, bestEpoch, minLoss, maxLoss = train_and_validation(num_epochs, num_samples, train_loader, validation_loader, optimizer, model, criterion, code_counter)

        #################    Loss_Epoch curve plot    ##################
        plot_error_curve(arr1, arr2, arr3, bestEpoch, minLoss, maxLoss, code_counter)
        print("Error Curve Saved !")

        #################    TESTING    ##################
        confusionmatrix, running_acc, count, running_loss, per_class_running_acc, per_class_count, map, valuess = test_data(code_counter, test_loader, num_samples, criterion)
        print("Testing Complete !")

        #################    TEST STATS    ##################
        print_test_stats_and_save_plots(confusionmatrix, running_acc, count, running_loss, per_class_running_acc, per_class_count, code_counter, map)
    elif options.lower() == 'b':
        #################    TESTING    ##################
        confusionmatrix, running_acc, count, running_loss, per_class_running_acc, per_class_count, map, valuess = test_data(code_counter, test_loader, num_samples, criterion)
        print("Testing Complete !")

        #################    TEST STATS    ##################
        print_test_stats_and_save_plots(confusionmatrix, running_acc, count, running_loss, per_class_running_acc, per_class_count, code_counter, map)
    else:
        print("Wrong Option - Only a and b work !")
    print("--- Process Done ---")



if __name__ == "__main__":
    main()
