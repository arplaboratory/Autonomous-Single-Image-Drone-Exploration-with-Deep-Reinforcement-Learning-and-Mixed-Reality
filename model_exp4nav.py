import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torchvision import models


class PPONetsMapRGB(nn.Module):
    def __init__(self,
                 act_dim,
                 device,
                 fix_cnn=False,
                 rnn_type='gru',
                 rnn_hidden_dim=128,
                 rnn_num=1,
                 use_rgb=True):
        super().__init__()
        # Input image size: [80, 80, 3] and [80, 80, 3]
        self.device = device
        self.rnn_type = rnn_type
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num = rnn_num
        self.use_rgb = use_rgb
        if self.use_rgb:
            self.rgb_resnet_model = models.resnet18(pretrained=True)
        self.large_map_resnet_model = models.resnet18(pretrained=True)
        self.small_map_resnet_model = models.resnet18(pretrained=True)
        resnet_models = [self.rgb_resnet_model,
                         self.large_map_resnet_model,
                         self.small_map_resnet_model] if self.use_rgb \
            else [self.large_map_resnet_model,
                  self.small_map_resnet_model]
        if fix_cnn:
            for model in resnet_models:
                for param in model:
                    param.requires_grad = False
        num_ftrs = self.large_map_resnet_model.fc.in_features
        num_in = 0
        if self.use_rgb:
            self.rgb_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
            self.rgb_resnet_model.fc = nn.Linear(num_ftrs, 128)
            num_in += 128
        self.large_map_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
        self.large_map_resnet_model.fc = nn.Linear(num_ftrs, 128)
        num_in += 128
        self.small_map_resnet_model.avgpool = nn.AvgPool2d(3, stride=1)
        self.small_map_resnet_model.fc = nn.Linear(num_ftrs, 128)
        num_in += 128

        self.merge_fc = nn.Linear(num_in, rnn_hidden_dim)
        if rnn_type == 'gru':
            rnn_cell = nn.GRU
        elif rnn_type == 'lstm':
            rnn_cell = nn.LSTM
        else:
            raise ValueError('unsupported rnn type: %s' % rnn_type)
        self.rnn = rnn_cell(input_size=rnn_hidden_dim,
                            hidden_size=rnn_hidden_dim,
                            num_layers=rnn_num)
        self.actor_fc = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 32),
            nn.ELU(),
        )
        self.actor_head = nn.Linear(32, act_dim)
        self.critic_fc = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 32),
            nn.ELU(),
        )
        self.critic_head = nn.Linear(32, 1)
        self.reset_parameters()
        print('========= requires_grad =========')
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        print('=================================')
        print('****************************')
        if self.use_rgb:
            print('RGB AND MAP as INPUT')
        else:
            print('MAP as INPUT')
        print('****************************')

    def forward(self, large_maps, small_maps, rgb_ims=None,
                hidden_state=None, action=None, deterministic=False):
        seq_len, batch_size, C, H, W = large_maps.size()
        large_maps = large_maps.view(batch_size * seq_len, C, H, W)
        l_cnn_out = self.large_map_resnet_model(large_maps)
        l_cnn_out = l_cnn_out.view(seq_len, batch_size, -1)

        seq_len, batch_size, C, H, W = small_maps.size()
        small_maps = small_maps.view(batch_size * seq_len, C, H, W)
        s_cnn_out = self.small_map_resnet_model(small_maps)
        s_cnn_out = s_cnn_out.view(seq_len, batch_size, -1)

        if self.use_rgb:
            seq_len, batch_size, C, H, W = rgb_ims.size()
            rgb_ims = rgb_ims.view(batch_size * seq_len, C, H, W)
            rgb_cnn_out = self.rgb_resnet_model(rgb_ims)
            rgb_cnn_out = rgb_cnn_out.view(seq_len, batch_size, -1)
            cnn_out = torch.cat((rgb_cnn_out, l_cnn_out, s_cnn_out), dim=-1)
        else:
            cnn_out = torch.cat((l_cnn_out, s_cnn_out), dim=-1)

        rnn_in = F.elu(self.merge_fc(cnn_out))

        rnn_out, hidden_state = self.rnn(rnn_in, hidden_state)
        pi = self.actor_head(self.actor_fc(rnn_out))
        val = self.critic_head(self.critic_fc(rnn_out))
        cat_dist = Categorical(logits=pi)
        if action is None:
            if not deterministic:
                action = cat_dist.sample()
            else:
                action = torch.max(pi, dim=2)[1]
        log_prob = cat_dist.log_prob(action)
        return action, log_prob, cat_dist.entropy(), val, hidden_state.detach(), pi

    def choose_action(self, input_data):
        self.eval()
        input_data['RGBCamera'] = torch.from_numpy(input_data['RGBCamera']).float().to(input_data['device'])
        input_data['RGBCamera'] = input_data['RGBCamera'].permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        # input_data['h_a'] = torch.from_numpy(input_data['h_a']).float().to(input_data['device'])  # TODO
        # input_data['h_c'] = torch.from_numpy(input_data['h_c']).float().to(input_data['device'])  # TODO
        input_data['track_h_a'] = torch.from_numpy(input_data['track_h_a']).float().to(input_data['device'])
        input_data['exp_h_a'] = torch.from_numpy(input_data['exp_h_a']).float().to(input_data['device'])
        input_data['fov_h_a'] = torch.from_numpy(input_data['fov_h_a']).float().to(input_data['device'])
        input_data['track_h_c'] = torch.from_numpy(input_data['track_h_c']).float().to(input_data['device'])
        input_data['exp_h_c'] = torch.from_numpy(input_data['exp_h_c']).float().to(input_data['device'])
        input_data['geo_map'] = torch.from_numpy(input_data['geo_map']).float().to(input_data['device']).unsqueeze(0).unsqueeze(0)

        action, log_prob, cat_dist, val, hidden_state, pi = self.forward(large_maps=input_data['geo_map'], small_maps=input_data['geo_map'], rgb_ims=input_data['RGBCamera'], hidden_state=input_data['exp_h_c'])

        # output_data = {'action_track': action_track.detach().cpu().numpy().squeeze(),
        #                'logits_track': logits_track.detach().squeeze().cpu().numpy(),
        #                'action_exp': action_exp.detach().cpu().numpy().squeeze(),
        #                'logits_exp': logits_exp.detach().squeeze().cpu().numpy(),
        #                'h_a_': forward_out['h_a_'], 'h_c_': forward_out['h_c_'],
        #                'fov': forward_out['fov']}  # TODO

        output_data = {'action_track': action.detach().cpu().numpy().squeeze(),
                       'logits_track': log_prob.detach().squeeze().cpu().numpy(),
                       'action_exp': action.detach().cpu().numpy().squeeze(),
                       'logits_exp': log_prob.detach().squeeze().cpu().numpy(),
                       'track_h_a_': hidden_state,
                       'exp_h_a_': hidden_state,
                       'fov_h_a_': torch.tensor(0),
                       'track_h_c_': hidden_state,
                       'exp_h_c_': hidden_state,
                       'fov': torch.tensor(0)}

        return output_data

    def init_hidden(self):
        batch_size = 1  # MODIFICATO ALE
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if self.rnn_type == 'lstm':
            return (torch.zeros(self.rnn_num,
                                batch_size,
                                self.rnn_hidden_dim).to(self.device),
                    torch.zeros(self.rnn_num,
                                batch_size,
                                self.rnn_hidden_dim).to(self.device))
        else:
            return torch.zeros(self.rnn_num,
                               batch_size,
                               self.rnn_hidden_dim).to(self.device)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias.data, 0)

    @property
    def learnable(self):
        return True
