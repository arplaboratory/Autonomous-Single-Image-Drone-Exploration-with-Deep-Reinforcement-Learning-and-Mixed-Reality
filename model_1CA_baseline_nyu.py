import torch
import torch.nn.functional as F
import torch.nn as nn
from model_ResNet import FeaturesExtractorResNet


class FOV_Net(nn.Module):
    def __init__(self, settings):
        super(FOV_Net, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']

        self.features_RGB = FeaturesExtractor_RGB(settings)
        self.RGB_shape = self.features_RGB.feat_shape

        # Linear
        self.lin1 = nn.Linear(self.RGB_shape, 750)
        self.lin2 = nn.Linear(750, 256)
        self.lin3 = nn.Linear(256, 128)
        self.output_fov = nn.Linear(128, 1)

    def forward(self, input_data):
        featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)

        # lin
        x1 = F.relu(self.lin1(featuresRGB))
        x2 = F.relu(self.lin2(x1))
        x3 = F.relu(self.lin3(x2))
        fov = self.output_fov(x3)
        # print(torch.sigmoid(fov))

        return fov


class FeaturesExtractor_RGB(nn.Module):
    def __init__(self, settings):
        super(FeaturesExtractor_RGB, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.width = settings['width']
        self.height = settings['height']
        self.d1 = 16
        self.d2 = 32

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=self.d1, kernel_size=8, stride=4)
        self.bnc1 = torch.nn.GroupNorm(int(self.d1 / 2), self.d1)
        self.conv2 = nn.Conv2d(in_channels=self.d1, out_channels=self.d2, kernel_size=4, stride=2)
        self.bnc2 = torch.nn.GroupNorm(int(self.d2 / 2), self.d2)

        self.feat_shape = 32 * 9 * 9

    def forward(self, input_data):
        inp = input_data['RGBCamera'].reshape(-1, self.channels, self.width, self.height)

        x1 = inp
        x2 = self.bnc1(F.relu(self.conv1(x1)))
        x3 = self.bnc2(F.relu(self.conv2(x2)))

        return x3

    @property
    def device(self):
        return next(self.parameters()).device


class ActorCriticIMPALA(nn.Module):
    def __init__(self, settings):
        super(ActorCriticIMPALA, self).__init__()
        self.settings = settings
        self.channels = settings['channels']
        self.d1 = settings['d1']
        self.d2 = settings['d2']
        self.d3 = settings['d3']
        self.dl = settings['dl']
        self.dr = settings['dr']

        self.features_RGB = FeaturesExtractor_RGB(settings)
        self.RGB_shape = self.features_RGB.feat_shape

        # Linear
        self.lin1 = nn.Linear(self.RGB_shape, 256)      # 3

        # GRU
        self.gru = nn.GRU(self.dl, 256)

        self.lin2 = nn.Linear(256, self.settings['output_a'] + 1)  # 3

    def forward(self, input_data):
        input_shape = input_data['RGBCamera'].shape
        input_ha = input_data['track_h_a']  # TODO input_ha = input_data['h_a']

        featuresRGB = self.features_RGB.forward(input_data).reshape(-1, self.RGB_shape)

        # lin
        x1 = F.relu(self.lin1(featuresRGB))

        # gru
        x_ = x1.view(input_shape[0], -1, 256)
        xgru, h_a_ = self.gru(x_, input_ha)

        out = self.lin2(xgru)
        logits_exp = out[:, :, :-1]
        values_exp = out[:, :, -1].unsqueeze(-1)

        return values_exp, values_exp, logits_exp, logits_exp, h_a_


# #####################################-Impala_Net-###################################################################
class IMPALA_Net(nn.Module):
    def __init__(self, settings):
        super(IMPALA_Net, self).__init__()
        self.settings = settings

        self.actor_critic = self.actor = self.critic = ActorCriticIMPALA(settings)
        self.fov_net = FOV_Net(settings)

    @property
    def learnable(self):
        return True

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_data):
        values_exp, values_track, logits_track, logits_exp, h_a_ = self.actor_critic(input_data)
        fov = self.fov_net(input_data)

        output_data = {'logits_track': logits_track,
                       'logits_exp': logits_exp,
                       'values_track': values_track,
                       'values_exp': values_exp,
                       'track_h_a_': h_a_, 'exp_h_a_': h_a_, 'fov_h_a_': h_a_,  # TODO 'h_a_': h_a_
                       'track_h_c_': h_a_,  # track_h_c_,
                       'exp_h_c_': h_a_,    # exp_h_c_,                    # TODO 'h_c_': h_c_
                       'fov': fov}

        return output_data

    def choose_action(self, input_data):
        self.eval()
        input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device']).unsqueeze(0)
        # input_data['h_a'] = torch.from_numpy(input_data['h_a']).float().to(input_data['device'])  # TODO
        # input_data['h_c'] = torch.from_numpy(input_data['h_c']).float().to(input_data['device'])  # TODO
        input_data['track_h_a'] = torch.from_numpy(input_data['track_h_a']).float().to(input_data['device'])
        input_data['exp_h_a'] = torch.from_numpy(input_data['exp_h_a']).float().to(input_data['device'])
        input_data['fov_h_a'] = torch.from_numpy(input_data['fov_h_a']).float().to(input_data['device'])
        input_data['track_h_c'] = torch.from_numpy(input_data['track_h_c']).float().to(input_data['device'])
        input_data['exp_h_c'] = torch.from_numpy(input_data['exp_h_c']).float().to(input_data['device'])

        forward_out = self.forward(input_data)

        logits_track = forward_out['logits_track']
        probs_track = torch.clamp(F.softmax(logits_track, dim=-1), 0.00001, 0.99999).data  # .values()
        m_track = torch.distributions.Categorical(probs_track)
        action_track = m_track.sample().type(torch.IntTensor)

        logits_exp = forward_out['logits_exp']
        probs_exp = torch.clamp(F.softmax(logits_exp, dim=-1), 0.00001, 0.99999).data  # .values()
        m_exp = torch.distributions.Categorical(probs_exp)
        action_exp = m_exp.sample().type(torch.IntTensor)

        # output_data = {'action_track': action_track.detach().cpu().numpy().squeeze(),
        #                'logits_track': logits_track.detach().squeeze().cpu().numpy(),
        #                'action_exp': action_exp.detach().cpu().numpy().squeeze(),
        #                'logits_exp': logits_exp.detach().squeeze().cpu().numpy(),
        #                'h_a_': forward_out['h_a_'], 'h_c_': forward_out['h_c_'],
        #                'fov': forward_out['fov']}  # TODO

        output_data = {'action_track': action_track.detach().cpu().numpy().squeeze(),
                       'logits_track': logits_track.detach().squeeze().cpu().numpy(),
                       'action_exp': action_exp.detach().cpu().numpy().squeeze(),
                       'logits_exp': logits_exp.detach().squeeze().cpu().numpy(),
                       'track_h_a_': forward_out['track_h_a_'],
                       'exp_h_a_': forward_out['exp_h_a_'],
                       'fov_h_a_': forward_out['fov_h_a_'],
                       'track_h_c_': forward_out['track_h_c_'],
                       'exp_h_c_': forward_out['exp_h_c_'],
                       'fov': forward_out['fov']}

        return output_data

    def get_values(self, input_data):
        self.eval()
        forward_out = self.forward(input_data)
        values_out = {'values_track': forward_out['values_track'],
                      'values_exp': forward_out['values_exp']}

        return values_out

    def init_hidden(self):
        h = torch.zeros((1, 1, self.settings['dr']), dtype=torch.float32)
        return h
######################################################################################################################
