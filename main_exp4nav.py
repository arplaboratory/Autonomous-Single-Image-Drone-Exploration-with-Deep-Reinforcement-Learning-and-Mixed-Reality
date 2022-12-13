def load_model(net_model, model_dir, device, step=None):
    if step is None:
        ckpt_file = os.path.join(model_dir, 'model_best.pth')
    else:
        ckpt_file = os.path.join(model_dir, 'ckpt_{:08d}.pth'.format(step))
    if not os.path.isfile(ckpt_file):
        raise ValueError("No checkpoint found at '{}'".format(ckpt_file))
    checkpoint = torch.load(ckpt_file)
    model_dict = net_model.state_dict()
    requires_grad_ori = {}
    for name, param in net_model.named_parameters():
        requires_grad_ori[name] = param.requires_grad
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net_model.load_state_dict(model_dict)
    global_iter = checkpoint['global_iter']
    #global_step = checkpoint['global_step']
    net_model.to(device)
    print('========= requires_grad =========')
    for name, param in net_model.named_parameters():
        param.requires_grad = requires_grad_ori[name]
        print(name, param.requires_grad)
    print('=================================')
    return global_iter


if __name__ == '__main__':
    from gym import spaces
    from remind.remind import Remind
    from model_DDPG import DDPG_Net
    # from model_ResNet import IMPALA_Net
    # from model_Encoder import IMPALA_Net
    # from model_LG import IMPALA_Net
    # from model1_LocMap import LocMap_Net
    # from model2_TrExp import IMPALA_Net
    # from model_Visibility import IMPALA_Net
    # from model_Asymmetric import IMPALA_Net

    # NYU
    #from model_1CA import IMPALA_Net       # MODELLO DEFINITIVO
    #from model_1CA_symmetric import IMPALA_Net
    from model_exp4nav import PPONetsMapRGB

    # from model_Symmetric import IMPALA_Net
    # from model_Switch import IMPALA_Net
    from learner import MyLearner
    from my_agent import MyAgent
    from live_plot import GridPlot
    from live_plot import RewardPlot
    from live_plot import MaprecPlot
    from memory_tracker import MemoryTracker
    from isarsocket.simple_socket import SimpleSocket
    import torch.multiprocessing as mp
    import numpy as np
    import random
    import torch
    import json
    import sys
    import os

    WANDB = False
    eval_mode = True
    TD3 = False
    IMPALA = True
    hot_reload = False
    TEST_LAB = False

    grid_plot = False
    reward_plot = False
    maprec_plot = False
    plot_list = [grid_plot, reward_plot, maprec_plot]
    n_plot = plot_list.count(True)

    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    np.random.seed(10)
    random.seed(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Path for Models
    pathname = os.path.dirname(sys.argv[0])
    abs_path = os.path.abspath(pathname)

    if IMPALA:
        print('IMPALA')
        if TEST_LAB:
            json_paths = [['{}/settings/IMPALA/settings_coord.json'.format(abs_path)],
                                     ['{}/settings/IMPALA/settings_coord.json'.format(abs_path),
                                        '{}/settings/IMPALA/settings_target.json'.format(
                                             abs_path)]]  # f = open('settings/IMPALA/settings.json')
            f = open('settings/IMPALA/settings_coord.json')  # TEST in LAB NYU
        else:
            json_paths = [['{}/settings/IMPALA/settings_exp4nav.json'.format(abs_path)],
                      ['{}/settings/IMPALA/settings_exp4nav.json'.format(abs_path), '{}/settings/IMPALA/settings_target.json'.format(abs_path)]]
            f = open('settings/IMPALA/settings_exp4nav.json')  # TRAIN
        #f = open('settings/IMPALA/settings.json')
        #json_paths = [['{}/settings/IMPALA/settings_coord.json'.format(abs_path)],
                      # f = open('settings/IMPALA/settings.json')
        #             ['{}/settings/IMPALA/settings_coord.json'.format(abs_path),
        #               '{}/settings/IMPALA/settings_target.json'.format(
        #                   abs_path)]]  # f = open('settings/IMPALA/settings.json')
        #f = open('settings/IMPALA/settings_coord.json')    # TEST in LAB NYU

        algorithm = 'IMPALA'
        net = PPONetsMapRGB
        n_actions = 6
        action_space = spaces.Discrete(n_actions)
        assert IMPALA != TD3

    else:
        print('DDPG')
        json_paths = ['{}/settings/DDPG/settings.json'.format(abs_path)]
        f = open('settings/IMPALA/settings.json')
        algorithm = 'DDPG'
        net = DDPG_Net
        action_space = [spaces.Box(low=-8, high=8, shape=(1,)), spaces.Box(low=-8, high=8, shape=(1,))]
        n_actions = 2

    json_settings = json.load(f)

    print('MAIN', os.getpid())

    mp.set_start_method('spawn')

    # [sequence, batch, features]
    seq_len = 50  # 50
    memory_len = 1500  # 10000
    episode_length = 1800  # 1800
    eval_episodes = 20
    N_Agents = 8
    batch_size = 24     # 24
    main_to_agent_address = 6734
    agent_address_list = [('127.0.0.1', main_to_agent_address + i) for i in range(N_Agents + 2)]
    plot_address = 10734

    # state_dict = 'best_model_2.pt'
    state_dict = 'best_model_7.pt'
    # state_dict = 'best_model_9.pt'
    # state_dict = 'best_model_3.pt'
    # state_dict = 'best_model_6.pt'
    # state_dict = 'best_model_1.pt'
    # state_dict = 'best_model_8.pt'
    # state_dict = 'best_model_4.pt'
    # state_dict = 'best_model_0.pt'
    # state_dict = 'best_model_5.pt'




    RGB_width = json_settings['sensor_settings']['RGBCamera']['width']
    RGB_height = json_settings['sensor_settings']['RGBCamera']['height']
    grid_cells = json_settings['sensor_settings']['OccupancyGrid']['size_x']

    settings = {
        'model_param': {
            'dl_1': 400,
            'dl_2': 300,
            'd1': 16,
            'd2': 32,
            'd3': 64,
            'dl': 256,
            'dr': 256,
            'channels': 3,
            'width': RGB_width,
            'height': RGB_height,
            'RGB_state_space': np.array([RGB_width, RGB_height, 3]),
            'state_space': 2,
            'init_w': 3e-3,
            'output_a': n_actions,
            'exp_distance': 50,
            'max_distance': 20,
            'TD3': TD3,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'grid_cells': grid_cells,
            'resnet': False,
        },
        'agent_gpu_id': 1,
        'update_steps': 100,
        'env_track': 'unrealisar-track-v0',
        'env_exp': 'unrealisar-exp-v0',
        'env_name': 'unrealisar-lg-v0',
        'multi_mod': True,
        'unreal': {
            'start_port_test': 9744,
            'start_port_train': 9736,
            'json_paths': json_paths,
            'render': True,
            'address': 'localhost'
        },
        'test': True,
        'fov_gt_test': False,
        'eval_mode': eval_mode,
        'eval_episodes': eval_episodes,
        'seq_len': seq_len,
        'memory_len': memory_len,
        'action_space': action_space,
        'action_space_noise': spaces.Box(low=-1, high=1, shape=(1,)),
        'WandB': WANDB,
        'workstation_name': 'Idra',
        'main_to_agent_address': ('127.0.0.1', main_to_agent_address),
        'first_address': ('127.0.0.1', 7734),
        'learner_address': ('127.0.0.1', 8734),
        'episode_length': episode_length,
        'warmup': seq_len + batch_size - 1 + int((seq_len + batch_size - 1) / episode_length) * (seq_len - 1),
        'on_policy': False,
        'learner_rank': 0,
        'lr_actor': 0.0001,
        'lr_critic': 0.001,
        'lr_impala': 0.0002,
        'batch_size': batch_size,
        'learner_gpu_id': 0,
        'memory_address': ('127.0.0.1', 7734 + N_Agents),
        'num_agents': N_Agents,
        'tau': 0.01,
        'gamma': 0.99,
        'RMSprop_eps': 0.1,
        'algorithm': algorithm,
        'baseline_cost': 0.5,
        'entropy_cost': 0.001,   # 0.001
        'navigation_cost': 1,
        'obstacle_rec_cost': 1,
        'tracker_rec_cost': 1e2,
        'target_rec_cost': 1e2,
        'TD3': TD3,
        'IMPALA': IMPALA,
        'update_settings': {
            'batch_size': batch_size,
            'min_new_samples': 50 * N_Agents,
            'mode': 'newest',
            'data_keys': {
                'RGBCamera': seq_len,
                'action_track': seq_len,
                'action_exp': seq_len,
                'reward_track': seq_len,
                'reward_exp': seq_len,
                'next_state': 1,
                'done_track': seq_len,
                'done_exp': seq_len,
                'track_h_a': 1,
                'exp_h_a': 1,
                'fov_h_a': 1,
                'track_h_c': 1,
                'exp_h_c': 1,
                'logits_track': seq_len,
                'logits_exp': seq_len,
                'yaw': seq_len,
                'ego_target': seq_len,
                'angle': seq_len,
                'distance': seq_len,
                'hit': seq_len,
                'fov_gt': seq_len,
                'bel': seq_len,
                'track_flag': seq_len
            }
        },
        'agents_fps_array': mp.Array('d', np.zeros(N_Agents)),
        'agent_rank': mp.Value('i', 0),
        'shared_counter': mp.Value('i', 0),
        'plot_param': {
            'plot_address': ('127.0.0.1', plot_address),
            'plot_list': plot_list,
            'n_plot': n_plot,
            'rew_len': episode_length,
            'eval_mode': eval_mode
        },
        'grid_cells': grid_cells,
        'obstacle_ch_weight': 1,
        'tracker_ch_weight': 1,
        'target_ch_weight': 2,
        'state_dict': state_dict
    }

    pids = []

    test_agent = MyAgent(settings)
    test_agent.start()
    pids.append(test_agent.pid)

    settings['test'] = False

    plots = []
    if grid_plot:
        grid_mp = GridPlot(settings=settings['plot_param'])
        plots.append(grid_mp)

    if reward_plot:
        reward_mp = RewardPlot(settings=settings['plot_param'])
        plots.append(reward_mp)

    if maprec_plot:
        maprec_mp = MaprecPlot(settings=settings['plot_param'])
        plots.append(maprec_mp)

    [plot_mp.start() for plot_mp in plots]
    [pids.append(plot_mp.pid) for plot_mp in plots]

    if not eval_mode:
        memory = Remind(first_address=('127.0.0.1', 7734), num_agents=N_Agents, seed=10)
        memory.start()
        pids.append(memory.pid)

        agents = [MyAgent(settings=settings) for i in range(N_Agents)]
        learner = MyLearner(settings=settings)

        [agent.start() for agent in agents]
        # [pids.append(agent.pid) for agent in agents]
        learner.start()
        pids.append(learner.pid)
        [pids.append(agent.pid) for agent in agents]

        in_socket = SimpleSocket(address_list=agent_address_list, server=True, name='main_to_agents')

        # Net
        gnet = net(settings=settings['model_param'])
        if hot_reload:
            #gnet.load_state_dict(torch.load('C:\\Users\\Idra\\Documents\\ActiveTracking\\NYU_models\\best_model_5.pt'))
            gnet.load_state_dict(torch.load('C:\\Users\\Idra\\Documents\\ActiveTracking\\NYU_models\\best_model_5.pt'))

        # test
        in_socket.server_send({'net': gnet}, socket_id=0)

        # train gnet
        for i in range(N_Agents):
            in_socket.server_send({'net': gnet}, socket_id=i + 1)

        # train learnable
        # for i in range(int(N_Agents / 2), N_Agents):
        #     in_socket.server_send(gnet, socket_id=i + 1)

        # learner
        in_socket.server_send({'net': gnet}, socket_id=N_Agents + 1)

        del gnet

        mem_tracker = MemoryTracker(pids)
        mem_tracker.start()

        [agent.join() for agent in agents]
        learner.join()
        memory.join()

    else:
        #gnet = net(settings=settings['model_param'])
        gnet = net(act_dim=6,
                  device=torch.device('cuda:0'),
                  fix_cnn=False,
                  rnn_type="gru",
                  rnn_hidden_dim=128,
                  rnn_num=1,
                  use_rgb=True)

        #gnet.load_state_dict(torch.load('C:\\Users\\Idra\\Documents\\ActiveTracking\\models\\' + state_dict))
        #gnet.load_state_dict(torch.load('C:\\Users\\Idra\\Documents\\ActiveTracking\\NYU_models\\second_model_right_left_f.pt'))
        #gnet.load_state_dict(torch.load('C:\\Users\\Idra\\Documents\\ActiveTracking\\NYU_models\\salvati\\first_model_in_lab.pt'))
        #gnet.load_state_dict(torch.load('C:\\Users\\Idra\\Documents\\ActiveTracking\\NYU_models\\salvati\\first_model_right_left.pt'))
        load_model(gnet, "C:\\Users\\Idra\\Documents\\ActiveTracking\\exp4nav_models\\pretrain\\pretrain\\il_map_rgb\\model", torch.device('cuda:0'))

        in_socket = SimpleSocket(address_list=[('127.0.0.1', main_to_agent_address)],
                                 server=True,
                                 name='main_to_agents')
        in_socket.server_send({'net': gnet}, socket_id=0)
        del gnet

        mem_tracker = MemoryTracker(pids)
        mem_tracker.start()

    [plot_mp.join() for plot_mp in plots]

    test_agent.join()

    mem_tracker.join()
