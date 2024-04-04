import torch
from torch import nn, optim
from models import cnn_discriminator, lstm_generator
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

prediction_period = 20   #num days input for generator
lr = .01
beta1 = .5
workers = 2
batch_size = 5
device = torch.device("cpu")
num_epochs = 50

def get_data(STOCK_NAMES):
    today_date = datetime.datetime.today()
    delta = datetime.timedelta(days= 800)
    starting_date = today_date - delta

    start_data = str(starting_date).split()[0]
    end_data = str(today_date).split()[0]
        
            
    stock_data = yf.download(STOCK_NAMES, start= start_data)
    return stock_data

def train_model(ticker_name, ticker_dataset):

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ticker_dataset.reshape(-1, 1))
    
    training_length = int(len(scaled_data) * .9)
    training_data = scaled_data[:training_length, :]
    testing_data = scaled_data[training_length :, :]

    x_train = []
    for i in range(training_length - prediction_period):
        x_train.append(training_data[i : i + prediction_period, :])
    x_test = []
    y_test = ticker_dataset[training_length + prediction_period : ]
    for j in range(len(testing_data) - prediction_period):
        x_test.append(testing_data[j : j + prediction_period, :])
    x_train, x_test, y_test = np.array(x_train).reshape(-1, prediction_period), np.array(x_test).reshape(-1, prediction_period), np.array(y_test)
    x_train, x_test = x_train[:, np.newaxis, :], x_test[:, np.newaxis, :]
    
    print(x_train.shape, len(x_train))
    print(x_test.shape, len(x_test))
    
    dataloader = torch.utils.data.DataLoader(x_train, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
    
    netD = cnn_discriminator(prediction_period, 1, 3, 2)
    netG = lstm_generator(prediction_period, batch_size, hidden_size = 64, output_size = 1)

    netD.to(device)
    netG.to(device)
    
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(prediction_period, device=device)
    
    real_label = 1.
    fake_label = 0.
    
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    gen_losses = []
    disc_losses = []

    #MAIN TRAINING LOOP

    for epoch in range(num_epochs):
        #batched loop through dataset
        for i, data in enumerate(dataloader, 0):
            #Update discriminator

            #first, train on all real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            bSize = real_cpu.size(0)

            label = torch.full((bSize,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(bSize, prediction_period, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
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
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            gen_losses.append(errG.item())
            disc_losses.append(errD.item())





STOCK_NAMES = ["GOOG"]
stock_data = get_data(STOCK_NAMES)


if len(STOCK_NAMES) > 1:
    for ticker in STOCK_NAMES:
        ticker_close_prices = stock_data['Close'][ticker].values
        print(ticker)
        train_model(ticker, ticker_close_prices)
else:
    ticker_close_prices = stock_data['Close'].values
    print(STOCK_NAMES[0])
    train_model(STOCK_NAMES[0], ticker_close_prices)