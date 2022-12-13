import torch
import torch.nn.functional as F
import torch.nn as nn


def create_res_block(dim):
    bnc1 = torch.nn.GroupNorm(int(dim / 2), dim)
    conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
    bnc2 = torch.nn.GroupNorm(int(dim / 2), dim)
    conv2 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1)
    return bnc1, conv1, bnc2, conv2


def create_big_block(in_dim, out_dim):
    bnc1 = torch.nn.GroupNorm(int(in_dim / 2), in_dim)
    conv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=0)
    max1 = nn.MaxPool2d(3, stride=2)

    bnc2, conv2, bnc3, conv3 = create_res_block(out_dim)
    bnc4, conv4, bnc5, conv5 = create_res_block(out_dim)
    return bnc1, bnc2, bnc3, bnc4, bnc5, conv1, conv2, conv3, conv4, conv5, max1


def res_block(x, conv1, bnc1, conv2, bnc2):
    x1 = conv1(F.relu(bnc1(x)))
    x1 = conv2(F.relu(bnc2(x1)))
    return x + x1


def big_block(x, conv, bnc, max_p):
    x1 = max_p(conv[0](bnc[0](x)))
    x1 = res_block(x1, conv[1], bnc[1], conv[2], bnc[2])
    x1 = res_block(x1, conv[3], bnc[3], conv[4], bnc[4])
    return x1


class FeaturesExtractorResNet(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractorResNet, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.dl = settings['dl']
        self.dr = settings['dr']
        self.feat_shape = 32 * 7 * 7

        # 1
        self.bnc1, self.bnc2, self.bnc3, self.bnc4, self.bnc5, \
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, \
            self.max1 = create_big_block(self.channels, self.d1)

        self.bnc_1 = [self.bnc1, self.bnc2, self.bnc3, self.bnc4, self.bnc5]
        self.conv_1 = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

        # 2
        self.bnc6, self.bnc7, self.bnc8, self.bnc9, self.bnc10, \
            self.conv6, self.conv7, self.conv8, self.conv9, self.conv10, \
            self.max2 = create_big_block(self.d1, self.d2)

        self.bnc_2 = [self.bnc6, self.bnc7, self.bnc8, self.bnc9, self.bnc10]
        self.conv_2 = [self.conv6, self.conv7, self.conv8, self.conv9, self.conv10]

        # 3
        self.bnc11, self.bnc12, self.bnc13, self.bnc14, self.bnc15, \
            self.conv11, self.conv12, self.conv13, self.conv14, self.conv15, \
            self.max3 = create_big_block(self.d2, self.d2)

        self.bnc_3 = [self.bnc11, self.bnc12, self.bnc13, self.bnc14, self.bnc15]
        self.conv_3 = [self.conv11, self.conv12, self.conv13, self.conv14, self.conv15]

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, 3, 84, 84)

        x1 = big_block(inp, self.conv_1, self.bnc_1, self.max1)
        x2 = big_block(x1, self.conv_2, self.bnc_2, self.max2)
        features = big_block(x2, self.conv_3, self.bnc_3, self.max3)

        return features


class ActorIMPALA(nn.Module):
    def __init__(self, settings):
        super(ActorIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.dl = settings['dl']
        self.dr = settings['dr']

        self.features = FeaturesExtractorResNet(settings)
        self.feat_shape = self.features.feat_shape

        # GroupNorm
        self.bn1 = nn.GroupNorm(self.d1, self.d2)

        # Linear
        self.lin = nn.Linear(self.feat_shape, self.dl)

        # GRU
        self.gru = nn.GRU(self.dl, self.dr)

        # hidden
        self.hidden = nn.Linear(self.dr, 100)

        # output
        self.output = nn.Linear(100, self.settings['output_a'])

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_ha = input_data['h_a']

        features = self.features.forward(input_data)

        x = F.relu(self.lin(self.bn1(F.relu(features)).view(-1, self.feat_shape)))
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, h_a_ = self.gru(x_, input_ha)

        xh = F.relu(self.hidden(xgru))

        logits = self.output(xh)

        return logits, h_a_


class CriticIMPALA(nn.Module):
    def __init__(self, settings):
        super(CriticIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.dl = settings['dl']
        self.dr = settings['dr']

        self.features = FeaturesExtractorResNet(settings)
        self.feat_shape = self.features.feat_shape

        # GroupNorm
        self.bn1 = nn.GroupNorm(self.d1, self.d2)

        # Linear
        self.lin = nn.Linear(self.feat_shape, self.dl)

        # GRU
        self.gru = nn.GRU(self.dl, self.dr)

        # hidden
        self.hidden = nn.Linear(self.dr, 100)

        # output
        self.output = nn.Linear(100, 1)

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_hc = input_data['h_c']

        features = self.features.forward(input_data)

        x = F.relu(self.lin(self.bn1(F.relu(features)).view(-1, self.feat_shape)))
        x_ = x.view(input_shape[0], -1, self.dl)
        xgru, h_c_ = self.gru(x_, input_hc)

        xh = F.relu(self.hidden(xgru))

        values = self.output(xh)

        return values, h_c_


class IMPALA_Net(nn.Module):
    def __init__(self, settings):
        super(IMPALA_Net, self).__init__()
        self.settings = settings
        self.actor = ActorIMPALA(settings)
        self.critic = CriticIMPALA(settings)

    @property
    def learnable(self):
        return True

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        logits, h_a_ = self.actor(input_data)
        values, h_c_ = self.critic(input_data)

        return logits, values, h_a_, h_c_

    def choose_action(self, input_data):
        self.eval()
        input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
        input_data['h_a'] = torch.from_numpy(input_data['h_a']).float().to(input_data['device'])
        input_data['h_c'] = torch.from_numpy(input_data['h_c']).float().to(input_data['device'])
        logits, _, h_a_, h_c_ = self.forward(input_data)

        probs = torch.clamp(F.softmax(logits, dim=-1), 0.00001, 0.99999).data  # .values()
        m = torch.distributions.Categorical(probs)
        action = m.sample().type(torch.IntTensor)
        output_data = {'action': action.detach().cpu().numpy().squeeze(), 'logits': logits, 'h_a_': h_a_,  'h_c_': h_c_}

        return output_data

    def get_logits(self, input_data):
        self.train()
        logits, _, _, _ = self.forward(input_data)

        return logits

    def get_values(self, input_data):
        self.eval()
        _, values, _, _ = self.forward(input_data)

        return values

    def init_hidden(self):
        h = torch.zeros((1, 1, self.settings['dl']), dtype=torch.float32)
        return h

