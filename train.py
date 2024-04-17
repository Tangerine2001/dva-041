import torch
from torch import nn, optim
from models import cnn_discriminator, lstm_generator
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from augment_data import augment_stockdata 

prediction_period = 20   #num days input for generator
predict_step = 3
lr = .0002
beta1 = .5
workers = 1
batch_size = 5
device = torch.device("cpu")
num_epochs = 100

def get_data(STOCK_NAMES):
    today_date = datetime.datetime.today()
    delta = datetime.timedelta(days= 1000)
    starting_date = today_date - delta

    start_data = str(starting_date).split()[0]
    end_data = str(today_date).split()[0]
        
            
    stock_data = yf.download(STOCK_NAMES, start= start_data)
    return stock_data

def train_model(ticker_dataset):

    aug_data = augment_stockdata(ticker_dataset)
    df = aug_data.add_ind()
    #print(df.head())

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(aug_data.add_sent_score())

    #print(scaled_data[:10, :])
    
    training_length = int(len(scaled_data) * .9)
    training_data = scaled_data[:training_length, :]
    testing_data = scaled_data[training_length :, :]

    x_train = []
    for i in range(training_length - prediction_period):
        x_train.append(training_data[i : i + prediction_period, :])
    x_test = []
    for j in range(len(testing_data) - prediction_period):
        x_test.append(testing_data[j : j + prediction_period, :])
    x_train, x_test = np.array(x_train), np.array(x_test)
    x_train, x_test = x_train[:, np.newaxis, :], x_test[:, np.newaxis, :]
    
    print(x_train.shape, len(x_train))
    print(x_test.shape, len(x_test))
    
    dataloaderTrain = torch.utils.data.DataLoader(x_train, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
    dataloaderTest = torch.utils.data.DataLoader(x_test, num_workers = workers)
    
    netD = cnn_discriminator(prediction_period, 1, 2, 1)
    netG = lstm_generator(prediction_period - predict_step, batch_size, hidden_size = 32, output_size = predict_step)

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
        for i, data in enumerate(dataloaderTrain, 0):
            #Update discriminator

            #first, train on all real batch
            netD.zero_grad()
            dataAlt = data.permute(1, 0, 2, 3)
            real_cpu = dataAlt[0].to(device)
            bSize = real_cpu.size(0)

            label = torch.full((bSize,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            #print(errD_real)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate fake last day with G
            exclude_last = data.permute(1, 0, 3, 2)[0].to(device)[:, :, : -1 * predict_step]
            #print(exclude_last.shape)
            fake = torch.concat((exclude_last, netG(exclude_last)), dim = 2)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach().permute(0, 2, 1)).view(-1)
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
            output = netD(fake.permute(0, 2, 1)).view(-1)
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
                    % (epoch, num_epochs, i, len(dataloaderTrain),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            gen_losses.append(errG.item())
            disc_losses.append(errD.item())

    for i, data in enumerate(dataloaderTest, 0):
        print(data.shape)
        last = data.permute(1, 0, 3, 2)[0].to(device)[:, :, -1 * predict_step :]
        lastP = netG(data.permute(1, 0, 3, 2)[0].to(device)[:, :, : -1 * predict_step])
        print("Actual : ", last.tolist()[0][3])
        print("Predicted : ", lastP.tolist()[0][3])




if __name__ == '__main__':
    STOCK_NAMES = ["GOOG"]
    stock_data = get_data(STOCK_NAMES)
    train_model(stock_data)