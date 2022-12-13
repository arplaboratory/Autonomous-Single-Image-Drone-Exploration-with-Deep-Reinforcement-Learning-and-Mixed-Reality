from isarsocket.simple_socket import SimpleSocket
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib
import numpy as np
import random
import math
import time
import cv2
import torch


def get_sockets(plot_address, name, num_agents=0):
    agents_out_address = [(plot_address[0], plot_address[1] + i) for i in range(num_agents + 1)]
    agents_out_socket = SimpleSocket(address_list=agents_out_address, server=False, name=name)
    return agents_out_socket


def get_plot_sockets(plot_address, plot_list):
    grid_plot_socket = SimpleSocket(address_list=[(plot_address[0], plot_address[1])], server=False,
                                    name='grid_socket') if plot_list[0] else None
    reward_plot_socket = SimpleSocket(address_list=[(plot_address[0], plot_address[1] + 1)], server=False,
                                      name='reward_socket') if plot_list[1] else None
    graph_plot_socket = SimpleSocket(address_list=[(plot_address[0], plot_address[1] + 2)], server=False,
                                     name='graph_socket') if plot_list[2] else None

    return grid_plot_socket, reward_plot_socket, graph_plot_socket


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # for QT and GTK
        f.canvas.manager.window.move(x, y)


class Plotter:
    def __init__(self, plot_width, plot_height, num_plot_values):
        self.width = plot_width
        self.height = plot_height
        self.color_list = [(255, 0, 0), (0, 250, 0), (0, 0, 250)]
        self.color = []
        self.val = []
        self.plot = np.ones((self.height, self.width, 3)) * 255

        for i in range(num_plot_values):
            self.color.append(self.color_list[i])

    # Update new values in plot
    def multiplot(self, val, label="plot"):
        self.val.append(val)
        while len(self.val) > self.width:
            self.val.pop(0)

        self.show_plot(label)

    # Show plot using opencv imshow
    def show_plot(self, label):
        self.plot = np.ones((self.height, self.width, 3)) * 255
        cv2.line(self.plot, (0, int(self.height / 2)), (self.width, int(self.height / 2)), (0, 255, 0), 1)
        for i in range(len(self.val) - 1):
            for j in range(len(self.val[0])):
                cv2.line(self.plot, (i, int(self.height / 2) - self.val[i][j]), (i + 1, int(self.height / 2) - self.val[i + 1][j]), self.color[j], 1)

        cv2.imshow(label, self.plot)
        cv2.waitKey(10)


class GridPlot(mp.Process):
    def __init__(self, settings):
        super(GridPlot, self).__init__()
        self.daemon = True

        self.settings = settings
        self.agent_out_socket = None

        self.fig, self.ax = plt.subplots(figsize=(5, 5), facecolor='lightblue')
        self.ax.set_title('Episode 1')
        self.cmap = colors.ListedColormap([(1, 1, 1), (1, 0, 0), (0, 0, 1)])
        self.data_stack = []

    def run(self):
        # self.agent_out_socket = SimpleSocket(address_list=[self.settings['plot_address']], server=False, name='grid_socket')
        # self.agent_out_socket, _, _ = get_plot_sockets(plot_address=self.settings['plot_address'], plot_list=self.settings['plot_list'])
        self.agent_out_socket = SimpleSocket(address_list=[(self.settings['plot_address'][0], self.settings['plot_address'][1])],
                                             server=False, name='grid_socket')

        ep = 1
        c = 0
        while True:
            received = self.agent_out_socket.client_receive()
            data = received[0]

            if len(self.data_stack) > 0:
                if data.reshape(1, -1).tolist() != self.data_stack[-1].reshape(1, -1).tolist():
                    plt.pcolor(data[::-1], cmap=self.cmap, edgecolors='k', linewidths=3)
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    plt.show(block=False)
                del self.data_stack[0]
            self.data_stack.append(data)

            if received[1]:
                ep += 1
                plt.close()
                self.fig, self.ax = plt.subplots(figsize=(5, 5), facecolor='lightblue')
                self.ax.set_title('Episode {}'.format(ep))
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                move_figure(self.fig, 0, 1200)
                plt.show(block=False)

            # self.agent_out_socket.client_send('ack')
            # print('PL', c)
            # c += 1


class RewardPlot(mp.Process):
    def __init__(self, settings):
        super(RewardPlot, self).__init__()
        self.daemon = True

        self.settings = settings
        self.agent_out_socket = None

        self.fig = plt.figure(2, figsize=(9, 2), edgecolor='k', facecolor='lightblue')
        self.ax = plt.axes()
        self.ax.grid(linewidth=0.5)
        self.ax.set_title('Episode 1')
        self.ax.set_ylabel('Reward')

        self.len_plot = 200

    def run(self):
        # self.agent_out_socket = SimpleSocket(address_list=[self.settings['plot_address']], server=False, name='rew_socket')
        # _, self.agent_out_socket, _ = get_plot_sockets(plot_address=self.settings['plot_address'], plot_list=self.settings['plot_list'])
        self.agent_out_socket = SimpleSocket(address_list=[(self.settings['plot_address'][0], self.settings['plot_address'][1] + 1)],
                                             server=False, name='grid_socket')
        # cnt = 0
        # nst = 0
        # ep = 1

        # plt.ylim(-0.03, 0.15)

        p = Plotter(100, 30, 1)
        while True:
            received = self.agent_out_socket.client_receive()
            v1 = int(np.ceil(received[0])) if received[0] > 0 else int(np.floor(received[0]))
            p.multiplot([v1], label='Reward')

            # if cnt % self.len_plot > int(self.len_plot/4):
            #     plt.xlim(cnt - int(self.len_plot/4), cnt + int(self.len_plot/10))
            # else:
            #     plt.xlim(self.len_plot * nst, self.len_plot * nst + int(self.len_plot/4))
            #
            # received = self.agent_out_socket.client_receive()
            #
            # self.ax.plot([cnt], [received[0]], 'bo')
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            # # move_figure(self.fig, 1200, 1200)
            # plt.show(block=False)
            #
            # if received[1] or (cnt % self.len_plot == 0 and cnt != 0):
            #     plt.close()
            #     self.fig = plt.figure(2, figsize=(9, 2), edgecolor='k', facecolor='lightblue')
            #     move_figure(self.fig, 1200, 1200)
            #     self.ax = plt.axes()
            #     self.ax.grid(linewidth=0.5)
            #     self.ax.set_title('Episode {}'.format(ep))
            #     self.ax.set_ylabel('Reward')
            #     nst += 1
            #     cnt = self.len_plot * nst
            #     if received[1]:
            #         ep += 1
            #         self.ax.set_title('Episode {}'.format(ep))
            #         cnt = 0
            #         nst = 0
            #
            # cnt += 1


class MaprecPlot(mp.Process):
    def __init__(self, settings):
        super(MaprecPlot, self).__init__()
        self.daemon = True

        self.settings = settings
        self.agent_out_socket = None

        # self.fig, self.ax = plt.subplots(figsize=(5, 5), facecolor='lightblue')

    def run(self):
        self.agent_out_socket = SimpleSocket(address_list=[(self.settings['plot_address'][0], self.settings['plot_address'][1] + 2)],
                                             server=False, name='grid_socket')

        while True:
            received = self.agent_out_socket.client_receive()
            plt.imshow(received.permute(1, 2, 0).detach().cpu().numpy())  # numpy float64 W, H, C
            plt.pause(1)  # 10
            # plt.close()

            # if len(self.data_stack) > 0:
            #     if data.reshape(1, -1).tolist() != self.data_stack[-1].reshape(1, -1).tolist():
            #         plt.pcolor(data[::-1], cmap=self.cmap, edgecolors='k', linewidths=3)
            #         self.fig.canvas.draw()
            #         self.fig.canvas.flush_events()
            #         plt.show(block=False)
            #     del self.data_stack[0]
            # self.data_stack.append(data)
            #
            # if received[1]:
            #     ep += 1
            #     plt.close()
            #     self.fig, self.ax = plt.subplots(figsize=(5, 5), facecolor='lightblue')
            #     self.ax.set_title('Episode {}'.format(ep))
            #     self.fig.canvas.draw()
            #     self.fig.canvas.flush_events()
            #     move_figure(self.fig, 0, 1200)
            #     plt.show(block=False)

# G = nx.DiGraph()
# def plot_graph(graph, info, done, episode, counter=0):
#     fig = plt.figure(1, figsize=(20, 20), dpi=80, edgecolor='k')
#
#     graph.add_node(info[-1])
#     if len(info) > 1:
#         graph.add_weighted_edges_from([(info[-2], info[-1], 1)])
#
#         if len(graph) > 2:
#             for i in range(1, len(info)):
#                 if info[-2] == info[i - 1] and info[-1] == info[i]:
#                     counter += 1
#                     graph[info[-2]][info[-1]]['weight'] = counter
#
#     if info[-2] == info[-1]:
#         graph[info[-2]][info[-1]]['weight'] += 1
#
#     pos = nx.kamada_kawai_layout(graph)
#
#     nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue', node_shape='o')
#     nx.draw_networkx_labels(graph, pos, font_size=9)
#     nx.draw_networkx_edges(graph, pos, width=3, edge_color='r', arrows=True, arrowsize=25)
#     nx.draw_networkx_edge_labels(graph, pos, font_size=11)
#
#     if done:
#         fig.savefig('plt/episode_' + str(episode)) if episode % 100 == 1 else 0
#         fig.clf()
#         graph.clear()
# plt.draw()
# plt.pause(0.001)
# plt.close()
# plot_graph(G, info['All_cells'], done, episodes)
# plt.close()
