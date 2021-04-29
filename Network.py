import torch.nn as nn
import torch.nn.functional as F
import torch

rnn_neuron_num = [1024, 1024]
encode_cnn_neuron_num = [16, 32, 64, 128, 256]
decode_cnn_neuron_num = [128, 64, 32, 16, 1]

EPSILON = 1e-10


class CRNN(nn.Module):
    """
    Input: [batch size, channels=1, T, n_fft]
    Output: [batch size, T, n_fft]
    """

    def __init__(self):
        super(CRNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=encode_cnn_neuron_num[0], kernel_size=(1, 3), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(num_features=encode_cnn_neuron_num[0])
        self.conv2 = nn.Conv2d(in_channels=encode_cnn_neuron_num[0], out_channels=encode_cnn_neuron_num[1], kernel_size=(1, 3), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(num_features=encode_cnn_neuron_num[1])
        self.conv3 = nn.Conv2d(in_channels=encode_cnn_neuron_num[1], out_channels=encode_cnn_neuron_num[2], kernel_size=(1, 3), stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(num_features=encode_cnn_neuron_num[2])
        self.conv4 = nn.Conv2d(in_channels=encode_cnn_neuron_num[2], out_channels=encode_cnn_neuron_num[3], kernel_size=(1, 3), stride=(1, 2))
        self.bn4 = nn.BatchNorm2d(num_features=encode_cnn_neuron_num[3])
        self.conv5 = nn.Conv2d(in_channels=encode_cnn_neuron_num[3], out_channels=encode_cnn_neuron_num[4], kernel_size=(1, 3), stride=(1, 2))
        self.bn5 = nn.BatchNorm2d(num_features=encode_cnn_neuron_num[4])

        self.GRU2 = nn.GRU(input_size=rnn_neuron_num[0], hidden_size=rnn_neuron_num[0], num_layers=2,
                             batch_first=True)

        self.convT1 = nn.ConvTranspose2d(in_channels=encode_cnn_neuron_num[4] * 2, out_channels=decode_cnn_neuron_num[0], kernel_size=(1, 3), stride=(1, 2))
        self.bnT1 = nn.BatchNorm2d(num_features=decode_cnn_neuron_num[0])
        self.convT2 = nn.ConvTranspose2d(in_channels=encode_cnn_neuron_num[3] * 2, out_channels=decode_cnn_neuron_num[1], kernel_size=(1, 3), stride=(1, 2))
        self.bnT2 = nn.BatchNorm2d(num_features=decode_cnn_neuron_num[1])
        self.convT3 = nn.ConvTranspose2d(in_channels=encode_cnn_neuron_num[2] * 2, out_channels=decode_cnn_neuron_num[2], kernel_size=(1, 3), stride=(1, 2))
        self.bnT3 = nn.BatchNorm2d(num_features=decode_cnn_neuron_num[2])
        # 
        self.convT4 = nn.ConvTranspose2d(in_channels=encode_cnn_neuron_num[1] * 2, out_channels=decode_cnn_neuron_num[3], kernel_size=(1, 3), stride=(1, 2), output_padding=(0, 1))
        self.bnT4 = nn.BatchNorm2d(num_features=decode_cnn_neuron_num[3])
        self.convT5 = nn.ConvTranspose2d(in_channels=encode_cnn_neuron_num[0] * 2, out_channels=decode_cnn_neuron_num[4], kernel_size=(1, 3), stride=(1, 2))
        self.bnT5 = nn.BatchNorm2d(num_features=decode_cnn_neuron_num[4])

    def forward(self, inputs):
        x = inputs 
        x1 = F.elu(self.bn1(self.conv1(x))) 
        x2 = F.elu(self.bn2(self.conv2(x1)))
        x3 = F.elu(self.bn3(self.conv3(x2))) 
        x4 = F.elu(self.bn4(self.conv4(x3))) 
        x5 = F.elu(self.bn5(self.conv5(x4))) 
        out5 = x5.permute(0, 2, 1, 3)
        out5 = out5.reshape(out5.size()[0], out5.size()[1], -1)
        gru, hn = self.GRU2(out5)
        output = gru.reshape(gru.size()[0], gru.size()[1], 256, -1)
        output = output.permute(0, 2, 1, 3)
        res = torch.cat((output, x5), 1)
        res1 = F.elu(self.bnT1(self.convT1(res))) 
        res1 = torch.cat((res1, x4), 1)
        res2 = F.elu(self.bnT2(self.convT2(res1)))  
        res2 = torch.cat((res2, x3), 1)
        res3 = F.elu(self.bnT3(self.convT3(res2)))  
        res3 = torch.cat((res3, x2), 1)
        res4 = F.elu(self.bnT4(self.convT4(res3)))
        res4 = torch.cat((res4, x1), 1)
        res5 = torch.sigmoid(self.convT5(res4)) 
        micro_amp = inputs[:, 0, :, :]
        est_clean_mag = torch.mul(res5.squeeze(), micro_amp.squeeze())
        return est_clean_mag

    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls()
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package