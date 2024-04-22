import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pickle
import datetime
from augment_data import augment_stockdata
from models import cnn_discriminator, lstm_generator, lstm_predictor
from plotUtil import plotter

prediction_period = 23   #num days input for generator
predict_step = 3

lr = .0002
plr = .001
beta1 = .5
workers = 1
batch_size = 5
pred_batch_size = 15
gen_hidden_size = 64
pred_hidden_size = 80

device = torch.device("cpu")

num_generated_data = 200

augmented_training = True
gen_train = True
sent_included = True

verbose = True

num_gan_epochs = 350 if gen_train else 0
num_pred_epochs = 150

def get_data(STOCK_NAMES):
    today_date = datetime.datetime.today()
    delta = datetime.timedelta(days= 1000)
    starting_date = today_date - delta

    start_data = str(starting_date).split()[0]
        
            
    stock_data = yf.download(STOCK_NAMES, start= start_data)
    return stock_data

def train_model(ticker_dataset, ticker):

    training_suffix = ""
    training_suffix += "SENT" if sent_included else ""
    training_suffix += "AUG" if sent_included else "REG"
    
    aug_data = augment_stockdata(ticker_dataset)

    df = aug_data.add_ind()
    if sent_included:
        df = aug_data.add_sent_score(ticker)
    #print(df.head())

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    num_feats = scaled_data.shape[1]
    #print(num_feats)
    
    windowed_data = []
    for i in range(len(scaled_data) - prediction_period):
        windowed_data.append(scaled_data[i : i + prediction_period, :])

    dataset = np.array(windowed_data)
    x_train, x_test = random_split(dataset, [0.9, 0.1])
    
    #print(x_train.shape, len(x_train))
    #print(x_test.shape, len(x_test))
    
    dataloaderTrain = DataLoader(x_train, batch_size = batch_size, shuffle = True, num_workers = workers)
    dataloaderTest = DataLoader(x_test, num_workers = workers)
    
    netD = cnn_discriminator(prediction_period, 1, 2, 1, sent = sent_included)
    netG = lstm_generator(num_feats, batch_size, hidden_size = 32, output_size =  num_feats)

    netD.to(device)
    netG.to(device)

    if not gen_train:
        netD.load_state_dict(torch.load("model_cache\discriminator" + training_suffix + "_" + ticker))
        netG.load_state_dict(torch.load("model_cache\generator" + training_suffix + "_" + ticker))
    
    criterion = nn.BCELoss()
    
    real_label = 1.
    fake_label = 0.
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    gen_losses = []
    disc_losses = []

    #MAIN TRAINING LOOP

    for epoch in range(num_gan_epochs):
        #batched loop through dataset
        for i, data in enumerate(dataloaderTrain, 0):
            #Update discriminator

            #first, train on all real batch
            real_cpu = data.to(device)
            bSize = real_cpu.size(0)

            #print(real_cpu.shape)

            label = torch.full((bSize,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)

            # Calculate loss on all-real batch
            netD.zero_grad()
            errD_real = criterion(output, label)
            #print(errD_real)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            #print(exclude_last.shape)
            noise = torch.randn((bSize, prediction_period, num_feats), device=device)
            fake = netG(noise)

            #print(fake.shape)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 100 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_gan_epochs, i, len(dataloaderTrain),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            gen_losses.append(errG.item())
            disc_losses.append(errD.item())
    if gen_train:
        glossplot = plotter("Generator losses for " + ticker, "loss", "Iteration", "BCE Loss", gen_losses)
        glossplot.get_plot(verbose= verbose)
        dlossplot = plotter("Discriminator losses for " + ticker, "loss", "Iteration", "BCE Loss", disc_losses)
        dlossplot.get_plot(verbose= verbose)

        torch.save(netD.state_dict(), "model_cache\discriminator" + training_suffix + "_" + ticker)
        torch.save(netG.state_dict(), "model_cache\generator" + training_suffix + "_" + ticker)

    if augmented_training:
        netG.eval()
        aug_data = None
        started = False
        for i in range(num_generated_data):
            new_noise = torch.randn((prediction_period, num_feats), device=device)
            genData = netG(new_noise)
            if not started:
                aug_data = genData.detach().numpy()[np.newaxis, :]
                started = True
            else:
                aug_data = np.concatenate((aug_data, genData.detach().numpy()[np.newaxis, :]), axis = 0)
        print(aug_data.shape)
        aug_x_train = np.concatenate((x_train, aug_data), axis = 0)
    else:
        aug_x_train = x_train

    aug_dataLoader = DataLoader(aug_x_train, batch_size = pred_batch_size, shuffle = True, num_workers = workers)
    

    pred_net = lstm_predictor(prediction_period - predict_step, num_feats, pred_batch_size, 32, predict_step)
    reg_loss = nn.MSELoss()

    pred_net.to(device)

    pred_optim = torch.optim.Adam(pred_net.parameters(), lr = plr)

    pred_losses = []

    for epoch in range(num_pred_epochs):
        for i, data in enumerate(aug_dataLoader, 0):
            #print(data.shape)
            model_input = data[:, : -1 * predict_step, :]
            actual = data[:, -1 * predict_step:, 3].float()
            predicted = pred_net(model_input).float()
            pred_net.zero_grad()
            errP = reg_loss(predicted, actual)
            errP.backward()
            pred_optim.step()
            if i % 100 == 0:
                    print('[%d/%d][%d/%d]\tLoss: %.4f\t'
                        % (epoch, num_pred_epochs, i, len(aug_dataLoader),
                            errP.item()))
            pred_losses.append(errP.item())
    plossplot = plotter("Predictor losses for " + ticker, "loss", "Iteration", "MSE Loss", pred_losses, log = True)
    plossplot.get_plot(verbose= verbose)
    pred_net.eval()
    numPlots = 3
    numCreated = 0
    for i, data in enumerate(dataloaderTest, 0):
        #print(data.shape)
        test_input = data[:, : -1 * predict_step, :]
        #print(test_input.shape)
        actual = scaler.inverse_transform(data[0, :, :])[:, 3]
        actualInput = actual[: -1 * predict_step]
        actualOut = actual[ -1 * predict_step:]
        predicted = scaler.inverse_transform(np.broadcast_to(pred_net(test_input).detach().numpy().reshape(-1, 1), (3, num_feats)))[:, 3]
        print("Predicted : ", predicted)
        print("Actual : ", actualOut)
        if numCreated < numPlots:
            plot = plotter("Predictor Forecasting " + ticker + " Price " + str(numCreated + 1), "Actual Input", "Day", "Price", actualInput, range(prediction_period)[: -1 * predict_step])
            plot.add_predicted("Actual Price", "blue", actualOut, range(prediction_period)[-1 * predict_step :])
            plot.add_predicted("Predicted Price", "orange", predicted, range(prediction_period)[-1 * predict_step :])
            plot.get_plot(verbose= verbose)
            numCreated += 1

    torch.save(pred_net.state_dict(), "model_cache\predictor" + training_suffix + "_" + ticker)
    with open('model_cache\scalerSENT_' + ticker + '.pkl' if sent_included else 'model_cache\scaler_' + ticker + '.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    

if __name__ == '__main__':
    STOCK_NAMES = ["GOOGL", "AMZN", "JNJ", "KO", "XOM", "IBM", "PFE", "PEP", "CVX"]
    for stock in STOCK_NAMES:
        stock_data = get_data([stock])
        train_model(stock_data, stock)