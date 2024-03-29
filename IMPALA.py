from utils import get_input_data_IMPALA
import numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.distributions
from adabelief_pytorch import AdaBelief

# import tensorflow as tf
# import vtrace_deepmind


class IMPALA:
    def __init__(self, settings):
        # Params
        self.settings = settings

        # GPU
        if torch.cuda.is_available() and self.settings['learner_gpu_id'] is not None:
            self.device = torch.device('cuda:' + str(self.settings['learner_gpu_id']))
        else:
            self.device = torch.device('cpu')

        # Networks  # TODO track+exp
        self.model = self.settings['net']['net'].to(self.device)
        # self.opt_actor_track = torch.optim.RMSprop(self.model.actor_track.parameters(), lr=self.settings['lr_impala'], eps=self.settings['RMSprop_eps'])
        # self.opt_actor_exp = torch.optim.RMSprop(self.model.actor_exp.parameters(), lr=self.settings['lr_impala'], eps=self.settings['RMSprop_eps'])
        # self.opt_fov_net = torch.optim.RMSprop(self.model.fov_net.parameters(), lr=self.settings['lr_impala'], eps=self.settings['RMSprop_eps'])
        # self.opt_critic_track = torch.optim.RMSprop(self.model.critic_track.parameters(), lr=self.settings['lr_impala'], eps=self.settings['RMSprop_eps'])
        # self.opt_critic_exp = torch.optim.RMSprop(self.model.critic_exp.parameters(), lr=self.settings['lr_impala'], eps=self.settings['RMSprop_eps'])
        self.opt_actor = torch.optim.RMSprop(self.model.actor.parameters(), lr=self.settings['lr_impala'], eps=self.settings['RMSprop_eps'])
        self.opt_critic = torch.optim.RMSprop(self.model.critic.parameters(), lr=self.settings['lr_impala'], eps=self.settings['RMSprop_eps'])
        self.fov_net = torch.optim.RMSprop(self.model.fov_net.parameters(), lr=self.settings['lr_impala'], eps=self.settings['RMSprop_eps'])

        # self.opt = [self.opt_actor_track, self.opt_actor_exp, self.opt_fov_net, self.opt_critic_track, self.opt_critic_exp]
        self.opt = [self.opt_actor, self.opt_critic, self.fov_net]

    def update(self, samples):
        input_data = get_input_data_IMPALA(samples, self.device)
        assert input_data['RGBCamera'].shape == (self.settings['seq_len'], self.settings['batch_size'],
                                                 self.settings['model_param']['RGB_state_space'][0],
                                                 self.settings['model_param']['RGB_state_space'][1],
                                                 self.settings['model_param']['RGB_state_space'][2])
        assert input_data['new_state'].shape == (self.settings['seq_len'], self.settings['batch_size'],
                                                 self.settings['model_param']['RGB_state_space'][0],
                                                 self.settings['model_param']['RGB_state_space'][1],
                                                 self.settings['model_param']['RGB_state_space'][2])
        assert input_data['done_track'].shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
        assert input_data['done_exp'].shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
        assert input_data['track_h_a'].shape == (1, self.settings['batch_size'], self.settings['model_param']['dr'])
        assert input_data['exp_h_a'].shape == (1, self.settings['batch_size'], self.settings['model_param']['dr'])
        assert input_data['fov_h_a'].shape == (1, self.settings['batch_size'], self.settings['model_param']['dr'])
        assert input_data['track_h_c'].shape == (1, self.settings['batch_size'], self.settings['model_param']['dr'])
        assert input_data['exp_h_c'].shape == (1, self.settings['batch_size'], self.settings['model_param']['dr'])
        assert input_data['logits_track'].shape == (self.settings['seq_len'], self.settings['batch_size'], self.settings['model_param']['output_a'])
        assert input_data['logits_exp'].shape == (self.settings['seq_len'], self.settings['batch_size'], self.settings['model_param']['output_a'])
        assert input_data['ego_target'].shape == (self.settings['seq_len'], self.settings['batch_size'], self.settings['model_param']['channels'], 11, 21)  # TODO 11, 21 settings
        assert input_data['fov_gt'].shape == (self.settings['seq_len'], self.settings['batch_size'], 1)
        assert input_data['track_flag'].shape == (self.settings['seq_len'], self.settings['batch_size'], 1)

        input_data['bel'] = input_data['bel'].reshape(-1, 3, self.settings['grid_cells'], self.settings['grid_cells'])
        input_data['ego_target'] = input_data['ego_target'].reshape(-1, self.settings['model_param']['channels'], 11, 21)

        # Masks
        flags = torch.nonzero(input_data['track_flag'])[:, :-1]
        select_inx_track = flags[flags[:, 0] == 0][:, 1]

        select_inx_exp = np.array(range(0, self.settings['batch_size']))
        select_inx_exp = np.delete(select_inx_exp, select_inx_track.cpu().numpy())
        select_inx_exp = torch.tensor(select_inx_exp, dtype=torch.long).to(self.device)

        # Masked Inputs
        input_data['logits_track'] = torch.index_select(input_data['logits_track'], 1, select_inx_track)
        input_data['action_track'] = torch.index_select(input_data['action_track'], 1, select_inx_track)
        input_data['reward_track'] = torch.index_select(input_data['reward_track'], 1, select_inx_track)
        input_data['done_track'] = torch.index_select(input_data['done_track'], 1, select_inx_track)

        input_data['logits_exp'] = torch.index_select(input_data['logits_exp'], 1, select_inx_exp)
        input_data['action_exp'] = torch.index_select(input_data['action_exp'], 1, select_inx_exp)
        input_data['reward_exp'] = torch.index_select(input_data['reward_exp'], 1, select_inx_exp)
        input_data['done_exp'] = torch.index_select(input_data['done_exp'], 1, select_inx_exp)

        self.model.train()
        output_data_te = self.model.forward(input_data)  # state

        # logits_track = output_data_te['logits_track']
        # logits_exp = output_data_te['logits_exp']
        # values_track = output_data_te['values_track'].squeeze()
        # values_exp = output_data_te['values_exp'].squeeze()

        logits_track = torch.index_select(output_data_te['logits_track'], 1, select_inx_track)
        logits_exp = torch.index_select(output_data_te['logits_exp'], 1, select_inx_exp)
        values_track = torch.index_select(output_data_te['values_track'].squeeze(), 1, select_inx_track)
        values_exp = torch.index_select(output_data_te['values_exp'].squeeze(), 1, select_inx_exp)

        # input_data['logits'] ---> behaviour_policy_logits
        # logits -----------------> target_policy_logits

        # assert logits_track.shape == (self.settings['seq_len'], self.settings['batch_size'], self.settings['model_param']['output_a'])
        # assert logits_exp.shape == (self.settings['seq_len'], self.settings['batch_size'], self.settings['model_param']['output_a'])
        # assert values_track.shape == (self.settings['seq_len'], self.settings['batch_size'])
        # assert values_exp.shape == (self.settings['seq_len'], self.settings['batch_size'])

        input_data['track_h_a'] = copy.copy(output_data_te['track_h_a_'])
        input_data['exp_h_a'] = copy.copy(output_data_te['exp_h_a_'])
        input_data['track_h_c'] = copy.copy(output_data_te['track_h_c_'])
        input_data['exp_h_c'] = copy.copy(output_data_te['exp_h_c_'])
        input_data['RGBCamera'] = copy.copy(input_data['new_state'])

        self.model.eval()
        # values_out = self.model.get_values(input_data)  # new_state
        # bootstrap_values_track = values_out['values_track']
        # bootstrap_values_track = bootstrap_values_track[-1] * (1 - input_data['done_track'][-1])
        # bootstrap_values_track.squeeze_()
        # bootstrap_values_exp = values_out['values_exp']
        # bootstrap_values_exp = bootstrap_values_exp[-1] * (1 - input_data['done_exp'][-1])
        # bootstrap_values_exp.squeeze_()

        values_out = self.model.get_values(input_data)  # new_state
        bootstrap_values_track = torch.index_select(values_out['values_track'], 1, select_inx_track)
        bootstrap_values_track = bootstrap_values_track[-1] * (1 - input_data['done_track'][-1])
        bootstrap_values_track.squeeze_()
        bootstrap_values_exp = torch.index_select(values_out['values_exp'], 1, select_inx_exp)
        bootstrap_values_exp = bootstrap_values_exp[-1] * (1 - input_data['done_exp'][-1])
        bootstrap_values_exp.squeeze_()

        # logits = torch.where(input_data['track_flag']==True, logits_track, logits_exp)
        # values = torch.where(input_data['track_flag']==True, values_track.unsqueeze(-1), values_exp.unsqueeze(-1)).squeeze()
        # bootstrap_values = torch.where(input_data['track_flag'][-1]==True, bootstrap_values_track.unsqueeze(-1), bootstrap_values_exp.unsqueeze(-1)).squeeze()
        # assert logits.shape == (self.settings['seq_len'], self.settings['batch_size'], self.settings['model_param']['output_a'])
        # assert values.shape == (self.settings['seq_len'], self.settings['batch_size'])

        probs_track = torch.clamp(F.softmax(logits_track, dim=-1), 0.000001, 0.999999)
        m_track = torch.distributions.Categorical(probs_track)
        probs_exp = torch.clamp(F.softmax(logits_exp, dim=-1), 0.000001, 0.999999)
        m_exp = torch.distributions.Categorical(probs_exp)

        discounts_track = (self.settings['gamma'] * (1 - input_data['done_track'])).squeeze()
        discounts_exp = (self.settings['gamma'] * (1 - input_data['done_exp'])).squeeze()

        # ###################################-deep_mind-shape-compatible-###############################################
        # v_trace  # TODO track+exp
        vs_track, pg_advantages_track = self.v_trace(probs_track.cpu(), input_data['logits_track'].cpu(), input_data['action_track'].cpu(),
                                        bootstrap_values_track.cpu(), values_track.cpu(), input_data['reward_track'].cpu(), discounts_track.cpu())

        vs_exp, pg_advantages_exp = self.v_trace(probs_exp.cpu(), input_data['logits_exp'].cpu(), input_data['action_exp'].cpu(),
                                         bootstrap_values_exp.cpu(), values_exp.cpu(), input_data['reward_exp'].cpu(), discounts_exp.cpu())

        # get_loss  # TODO track+exp
        policy_loss_track, value_loss_track, entropy_loss_track, impala_loss_track = \
            self.get_loss(input_data['action_track'], pg_advantages_track, m_track, vs_track, values_track, probs_track)

        policy_loss_exp, value_loss_exp, entropy_loss_exp, impala_loss_exp = \
            self.get_loss(input_data['action_exp'], pg_advantages_exp, m_exp, vs_exp, values_exp, probs_exp)
        # ##############################################################################################################

        # ##########################################-auxiliary_loss-####################################################
        fov_loss = self.fov_bce(output_data_te['fov'], input_data['fov_gt'])  # TODO track+exp
        # ##############################################################################################################

        # get_grad
        # _ = self.get_grad(self.model.actor_track.named_parameters())  # TODO track+exp
        # _ = self.get_grad(self.model.actor_exp.named_parameters())
        # _ = self.get_grad(self.model.fov_net.named_parameters())
        # _ = self.get_grad(self.model.critic_track.named_parameters())
        # _ = self.get_grad(self.model.critic_exp.named_parameters())
        # _ = self.get_grad(self.model.named_parameters())
        _ = self.get_grad(self.model.actor.named_parameters())
        _ = self.get_grad(self.model.critic.named_parameters())
        _ = self.get_grad(self.model.fov_net.named_parameters())
        _ = self.get_grad(self.model.named_parameters())

        # backpropagation
        self.model.zero_grad()
        impala_loss_track.backward(retain_graph=True)  # TODO track+exp
        impala_loss_exp.backward(retain_graph=True)
        fov_loss.backward()
        # impala_loss_exp.backward()

        # torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), 100)
        # torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(), 100)

        # get_grad
        # grad_norm_policy_track = self.get_grad(self.model.actor_track.named_parameters())  # TODO track+exp
        # grad_norm_policy_exp = self.get_grad(self.model.actor_exp.named_parameters())
        # grad_norm_fov = self.get_grad(self.model.fov_net.named_parameters())
        # grad_norm_critic_track = self.get_grad(self.model.critic_track.named_parameters())
        # grad_norm_critic_exp = self.get_grad(self.model.critic_exp.named_parameters())
        # grad_norm_overall = self.get_grad(self.model.named_parameters())
        grad_norm_actor = self.get_grad(self.model.actor.named_parameters())
        grad_norm_critic = self.get_grad(self.model.critic.named_parameters())
        grad_norm_fov = self.get_grad(self.model.fov_net.named_parameters())
        grad_norm_overall = self.get_grad(self.model.named_parameters())

        # optimizer-step
        self.opt[0].step()
        self.opt[1].step()
        self.opt[2].step()
        # self.opt[3].step()  # TODO track+exp
        # self.opt[4].step()

        policy_loss_track = policy_loss_track.detach().cpu().numpy()  # TODO track+exp
        policy_loss_exp = policy_loss_exp.detach().cpu().numpy()
        critic_loss_track = value_loss_track.detach().cpu().numpy()
        critic_loss_exp = value_loss_exp.detach().cpu().numpy()
        entropy_loss_track = entropy_loss_track.detach().cpu().numpy()
        entropy_loss_exp = entropy_loss_exp.detach().cpu().numpy()
        fov_loss = fov_loss.detach().cpu().numpy()
        impala_loss_track = impala_loss_track.detach().cpu().numpy()
        impala_loss_exp = impala_loss_exp.detach().cpu().numpy()

        # if not type(grad_norm_policy_track) == float:  # TODO track+exp
        #     grad_norm_policy_track = grad_norm_policy_track.cpu()
        # if not type(grad_norm_policy_exp) == float:
        #     grad_norm_policy_exp = grad_norm_policy_exp.cpu()
        if not type(grad_norm_fov) == float:
            grad_norm_fov = grad_norm_fov.cpu()
        # if not type(grad_norm_critic_track) == float:
        #     grad_norm_critic_track = grad_norm_critic_track.cpu()
        # if not type(grad_norm_critic_exp) == float:
        #     grad_norm_critic_exp = grad_norm_critic_exp.cpu()
        if not type(grad_norm_actor) == float:
            grad_norm_actor = grad_norm_actor.cpu()
        if not type(grad_norm_critic) == float:
            grad_norm_critic = grad_norm_critic.cpu()
        if not type(grad_norm_overall) == float:
            grad_norm_overall = grad_norm_overall.cpu()

        # grad_norm_policy_track = np.array(grad_norm_policy_track, dtype=np.float32)  # TODO track+exp
        # grad_norm_policy_exp = np.array(grad_norm_policy_exp, dtype=np.float32)
        grad_norm_fov = np.array(grad_norm_fov, dtype=np.float32)
        # grad_norm_critic_track = np.array(grad_norm_critic_track, dtype=np.float32)
        # grad_norm_critic_exp = np.array(grad_norm_critic_exp, dtype=np.float32)
        grad_norm_actor = np.array(grad_norm_actor, dtype=np.float32)
        grad_norm_critic = np.array(grad_norm_critic, dtype=np.float32)
        grad_norm_overall = np.array(grad_norm_overall, dtype=np.float32)

        output_data = {
            'policy_loss_track': policy_loss_track,  # TODO track+exp
            'policy_loss_exp': policy_loss_exp,
            'critic_loss_track': critic_loss_track,
            'critic_loss_exp': critic_loss_exp,
            'entropy_loss_track': entropy_loss_track,
            'entropy_loss_exp': entropy_loss_exp,
            'fov_loss': fov_loss,
            'overall_loss_track': impala_loss_track,
            'overall_loss_exp': impala_loss_exp,

            # 'grad_norm_policy_track': grad_norm_policy_track,  # TODO track+exp
            # 'grad_norm_policy_exp': grad_norm_policy_exp,
            'grad_norm_fov': grad_norm_fov,
            # 'grad_norm_critic_track': grad_norm_critic_track,
            # 'grad_norm_critic_exp': grad_norm_critic_exp,
            'grad_norm_policy': grad_norm_actor,
            'grad_norm_critic': grad_norm_critic,
            'grad_norm_overall': grad_norm_overall
        }

        return output_data

    def v_trace(self, probs, bl, ba, bootstrapped_value, values, br, discounts):
        # assert probs.shape == (self.settings['seq_len'], self.settings['batch_size'], self.settings['model_param']['output_a'])
        # assert bl.shape == (self.settings['seq_len'], self.settings['batch_size'], self.settings['model_param']['output_a'])
        # assert ba.shape == (self.settings['seq_len'], self.settings['batch_size'])
        # assert bootstrapped_value.shape == torch.Size([self.settings['batch_size']])
        # assert values.shape == (self.settings['seq_len'], self.settings['batch_size'])
        # assert br.shape == (self.settings['seq_len'], self.settings['batch_size'])
        # assert discounts.shape == (self.settings['seq_len'], self.settings['batch_size'])

        m = torch.distributions.Categorical(probs)
        rho_threshold = 1
        pg_rho_threshold = 1
        b_probs = torch.clamp(F.softmax(bl, dim=-1), 0.000001, 0.999999)
        b_m = torch.distributions.Categorical(b_probs)

        target_action_log_probs = m.log_prob(ba)  # pi
        behaviour_action_log_prob = b_m.log_prob(ba)  # mi

        log_rhos = target_action_log_probs - behaviour_action_log_prob
        rhos = torch.exp(log_rhos)

        clipped_rhos = torch.clamp(rhos, 0, rho_threshold)
        clipped_pg_rhos = torch.clamp(rhos, 0, pg_rho_threshold)

        bootstrapped_value.unsqueeze_(0)
        values_t_plus_1 = torch.cat((values[1:], bootstrapped_value), dim=0)

        deltas = clipped_rhos * (br + discounts * values_t_plus_1 - values)

        acc = 0
        dt = []

        # Bellman -> vs: ground truth, values: critic estimation
        for i in reversed(range(len(deltas))):
            acc = deltas[i] + discounts[i] * clipped_rhos[i] * acc
            dt.append(acc)

        vs_minus_v_xs = torch.stack(dt).flip(0)
        vs = (vs_minus_v_xs + values)

        vs_t_plus_1 = torch.cat((vs[1:], bootstrapped_value), dim=0)
        pg_advantages = clipped_pg_rhos * (br + discounts * vs_t_plus_1 - values)  # delta_t_V

        # no backpropagation though returned values
        return vs.detach().to(self.device), pg_advantages.detach().to(self.device)

    # def get_loss(self, ba, pg_advantages, m, vs, values, probs, track_flag):
    #     assert pg_advantages.shape == (self.settings['seq_len'], self.settings['batch_size'])
    #     assert vs.shape == (self.settings['seq_len'], self.settings['batch_size'])
    #     tmp = -m.log_prob(ba)
    #     assert tmp.shape == pg_advantages.shape
    #     assert vs.shape == values.shape
    #     policy_loss = (-m.log_prob(ba) * pg_advantages).sum()
    #     value_loss = 0.5 * (vs - values).pow(2).sum()
    #     probs_track = torch.where(track_flag == True, probs, torch.tensor([-1.]).to(self.device))
    #     probs_track = probs_track[probs_track != -1.]
    #     probs_exp = torch.where(track_flag == False, probs, torch.tensor([-1.]).to(self.device))
    #     probs_exp = probs_exp[probs_exp != -1.]
    #     entropy_loss_track = (probs_track * - torch.log(probs_track)).sum()
    #     entropy_loss_exp = (probs_exp * - torch.log(probs_exp)).sum()
    #     loss = policy_loss + self.settings['baseline_cost'] * value_loss - \
    #            (self.settings['entropy_cost'] * entropy_loss_track + self.settings['entropy_cost'] * entropy_loss_exp)
    #
    #     return policy_loss.to(self.device), value_loss.to(self.device), \
    #            entropy_loss_track.to(self.device), entropy_loss_exp.to(self.device), loss.to(self.device)

    def get_loss(self, ba, pg_advantages, m, vs, values, probs):
        # assert pg_advantages.shape == (self.settings['seq_len'], self.settings['batch_size'])
        # assert vs.shape == (self.settings['seq_len'], self.settings['batch_size'])
        tmp = -m.log_prob(ba)
        assert tmp.shape == pg_advantages.shape
        assert vs.shape == values.shape
        policy_loss = (-m.log_prob(ba) * pg_advantages).sum()
        value_loss = 0.5 * (vs - values).pow(2).sum()
        entropy_loss = (probs * - torch.log(probs)).sum()
        loss = policy_loss + self.settings['baseline_cost'] * value_loss - self.settings['entropy_cost'] * entropy_loss

        return policy_loss.to(self.device), value_loss.to(self.device), \
               entropy_loss.to(self.device), loss.to(self.device)

    def cross_entropy(self, input, target, ch_weight=1):
        """ 1-channel Cross Entropy """
        input = torch.clamp(input, min=0., max=1.)
        assert input.shape == target.shape and \
               input.shape == (self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])
        i = input.reshape(len(input), -1)
        t = target.reshape(len(target), -1)
        l_out = []
        output = 0
        for cnt in range(len(i)):  # loop needed to compute new set of weights
            index = torch.where(t[cnt] > 0)
            if index[-1].nelement() == 0:  # handle no target location
                continue
            index = index[-1][int((len(index[-1])) / 2)].unsqueeze(0)  # handle multiple target locations
            w = torch.Tensor([1]*(len(i[-1]))).to(self.device)  # weighting target location 🚀
            w[index] = ch_weight
            loss = torch.nn.CrossEntropyLoss(weight=w, reduction='none')
            l_out.append(loss(i[cnt, :].unsqueeze(0), index))
            output = (sum(l_out) / len(l_out))
            assert output.nelement() == 1
        return output

    def mse_ego(self, input, target):
        """ 1-channel mean squared error """
        input = torch.clamp(input, min=0., max=1.)
        assert input.shape == target.shape and \
               input.shape == (self.settings['seq_len'] * self.settings['batch_size'], 11, 21)
        diff = input - target
        assert diff.shape == (self.settings['seq_len'] * self.settings['batch_size'], 11, 21)
        err = torch.mean(diff ** 2)
        assert err.shape == torch.Size([])
        return err

    def mse(self, input, target):
        """ 1-channel mean squared error """
        input = torch.clamp(input, min=0., max=1.)
        assert input.shape == target.shape and \
               input.shape == (self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])
        diff = input - target
        assert diff.shape == (self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])
        err = torch.mean(diff ** 2)
        assert err.shape == torch.Size([])
        return err

    def KLdiv(self, input, target):
        """ 1-channel KL divergence """
        assert input.shape == target.shape and \
               input.shape == (self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])
        probs = F.softmax(input.reshape(self.settings['seq_len'] * self.settings['batch_size'], -1), dim=-1)
        assert torch.all(torch.round(torch.sum(probs, dim=-1)) == 1.)
        probs = probs.reshape(self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])
        output = F.kl_div(torch.log(probs), target, reduction='sum')
        assert output.shape == torch.Size([])
        return output

    def bce(self, input, target):
        """ 1-channel binary cross entropy """
        with torch.autograd.set_detect_anomaly(True):
            loss = torch.nn.BCELoss()
            s = torch.nn.Sigmoid()
            assert input.shape == target.shape and \
                   input.shape == (self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])
            input = s(input)
            output = loss(input, target)
            assert output.shape == torch.Size([])
        return output

    def fov_bce(self, input, target):
        """ 1-channel binary cross entropy """
        input = input.reshape(self.settings['seq_len'] * self.settings['batch_size'], 1)
        target = target.reshape(self.settings['seq_len'] * self.settings['batch_size'], 1)
        loss = torch.nn.BCELoss()
        s = torch.nn.Sigmoid()
        input = s(input)
        output = loss(input, target)
        assert output.shape == torch.Size([])
        return output

    def w_bce(self, input, target):
        """ 1-channel binary cross entropy """
        assert input.shape == target.shape and \
               input.shape == (self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])

        weight = torch.tensor([1, self.settings['grid_cells'] * self.settings['grid_cells']]).to(self.device)
        weight_ = weight[target.data.view(-1).long()].view_as(target)
        loss = torch.nn.BCELoss(reduction='none')
        s = torch.nn.Sigmoid()
        input = s(input)
        output = loss(input, target)
        output *= weight_.float()
        output = torch.mean(output)
        assert output.shape == torch.Size([])
        return output

    @staticmethod
    def get_grad(params):
        grad_norm = 0.0
        for name1, net1 in params:
            if net1.grad is None:
                pass        # print("Gradients None -> Name: ", name1)
            else:
                grad_norm += net1.grad.pow(2).sum()
        grad_norm = grad_norm ** (1 / 2)

        return grad_norm

    # #############################-deep-mind_v-trace_and_loss-######################################################
    # input_data['logits'] = input_data['logits'].detach().numpy()
    # input_data['action'] = input_data['action'].detach().numpy().astype('int32')
    # input_data['reward'] = input_data['reward'].detach().numpy()
    # logits = logits.detach().numpy()
    # discounts = discounts.detach().numpy()
    # values = values.detach().numpy()
    # bootstrap_values = bootstrap_values.detach().numpy()
    #
    # vtrace_returns = vtrace_deepmind.from_logits(behaviour_policy_logits=input_data['logits'],
    #                                              target_policy_logits=logits,
    #                                              actions=input_data['action'],
    #                                              discounts=discounts,
    #                                              rewards=input_data['reward'],
    #                                              values=values,
    #                                              bootstrap_value=bootstrap_values,
    #                                              clip_rho_threshold=1.0,
    #                                              clip_pg_rho_threshold=1.0,
    #                                              name='vtrace-from_logits')
    #
    # overall_loss = vtrace_deepmind.compute_policy_gradient_loss(logits,
    #                                                             input_data['action'],
    #                                                             vtrace_returns[1])
    # overall_loss += 0.5 * vtrace_deepmind.compute_baseline_loss(vtrace_returns[0] - values)
    # overall_loss += 0.00025 * vtrace_deepmind.compute_entropy_loss(logits)
    # ################################################################################################################

    # @staticmethod
    # # TODO
    # def importance_weighting(inp, tar, index, w, win_size):
    #     with torch.autograd.set_detect_anomaly(True):
    #         # position weighting
    #         inp[index] *= w
    #         # zero-pad
    #         pad = int((win_size - 1) / 2)
    #         inp = F.pad(input=inp, pad=[pad, pad, pad, pad], mode='constant', value=0)
    #         # window weighting over seq_len * batch imgs
    #         for img in range(len(inp)):
    #             # win_size = 3
    #             ring1 = []
    #             for i in range(index[1][img] - 1, (index[1][img] + 1) + 1):
    #                 for j in range(index[2][img] - 1, (index[2][img] + 1) + 1):
    #                     if (i, j) != (index[1][img], index[2][img]):
    #                         inp[img, i + pad, j + pad] *= (w / 2.5)
    #                         ring1.append((i, j))
    #             # win_size = 5
    #             if win_size >= 5:
    #                 ring2 = []
    #                 for i in range(index[1][img] - 2, (index[1][img] + 2) + 1):
    #                     for j in range(index[2][img] - 2, (index[2][img] + 2) + 1):
    #                         if (i, j) != (index[1][img], index[2][img]) and (i, j) not in ring1:
    #                             inp[img, i + pad, j + pad] *= (w / 5)
    #                             ring2.append((i, j))
    #                 # win_size = 7
    #                 if win_size >= 7:
    #                     ring3 = []
    #                     for i in range(index[1][img] - 3, (index[1][img] + 3) + 1):
    #                         for j in range(index[2][img] - 3, (index[2][img] + 3) + 1):
    #                             if (i, j) != (index[1][img], index[2][img]) and (i, j) not in ring2 and (i, j) not in ring1:
    #                                 inp[img, i + pad, j + pad] *= (w / 10)
    #                                 ring3.append((i, j))
    #         # un-pad
    #         inp_weighted = copy.copy(inp[:, pad:-pad, pad:-pad])
    #         # position weighting
    #         tar[index] *= (w * 2)
    #
    #     return inp_weighted, tar
    #
    #
    # def mse_tracker(self, input, target, weight, window_size=3):
    #     assert input.shape == target.shape and \
    #            input.shape == (self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])
    #     index = torch.where(target > 0)
    #     assert len(index[0]) == len(target)
    #     input, target = self.importance_weighting(inp=input, tar=target, index=index, w=weight, win_size=window_size)
    #     diff = target - input
    #     assert diff.shape == (self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])
    #     err = torch.mean(diff ** 2)
    #     assert err.shape == torch.Size([])
    #     print(input[-1])
    #     print(target[-1])
    #     return err

    # def cross_entropy(self, input, target):
    #     loss = torch.nn.NLLLoss()
    #     assert input.shape == target.shape and \
    #            input.shape == (self.settings['seq_len'] * self.settings['batch_size'],
    #                            self.settings['model_param']['channels'], self.settings['grid_cells'], self.settings['grid_cells'])
    #     probs = F.softmax(input, dim=1)  # pixel-wise softmax
    #     assert torch.all(torch.round(torch.sum(probs, dim=1)) == 1.)  # probs sum to 1
    #     log_prob = torch.log(probs)
    #     target = self.inv_RGB(target)
    #     output = loss(log_prob, target)
    #     assert output.shape == torch.Size([])
    #     return output
    #
    # def inv_RGB(self, target):  # TODO
    #     tmp = (torch.ones((self.settings['seq_len'] * self.settings['batch_size'],
    #                        self.settings['grid_cells'], self.settings['grid_cells'])) * 2).long().to(self.device)
    #     for img in range(len(target)):
    #         for i in range(len(target[0, 0])):
    #             for j in range(len(target[0, 0])):
    #                 if target[img, 0, i, j] == 1:
    #                     tmp[img, i, j] = 0
    #                 elif target[img, 1, i, j] == 1:
    #                     tmp[img, i, j] = 1
    #                 elif target[img, 2, i, j] == 1:
    #                     tmp[img, i, j] = 2
    #                 else:
    #                     tmp[img, i, j] = 2
    #     return tmp

    # def mae(self, imageA, imageB):
    #     """ 1-channel mean absolute error """
    #     assert imageA.shape == imageB.shape and \
    #            imageA.shape == (self.settings['seq_len'] * self.settings['batch_size'], self.settings['grid_cells'], self.settings['grid_cells'])
    #     diff = imageA - imageB
    #     assert diff.shape == (self.settings['seq_len'] * self.settings['batch_size'],
    #                           self.settings['model_param']['channels'], self.settings['grid_cells'], self.settings['grid_cells'])
    #     err = torch.mean(torch.abs(diff))
    #     return err
    #
    # def psnr_loss(self, imageA, imageB):
    #     assert imageA.shape == imageB.shape and \
    #            imageA.shape == (self.settings['seq_len'] * self.settings['batch_size'],
    #                             self.settings['model_param']['channels'], self.settings['grid_cells'], self.settings['grid_cells'])
    #     """Peak Signal-to-Noise Ratio: ratio between the maximum possible power
    #     of a signal and the power of corrupting noise that affects the fidelity of its representation"""
    #     mse = torch.mean((imageA - imageB) ** 2)
    #     psnr = 20 * torch.log10(1. / torch.sqrt(mse))
    #     output = psnr ** (-1)
    #     assert output.shape == torch.Size([])
    #     return output

