import copy
import math
import numpy as np
import matplotlib.cm as cmx
# import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Wedge
from matplotlib.patches import ConnectionPatch
from math import sin, cos, atan2, sqrt, pow
from mamp.tools import rvo_math
from mamp.tools.vector import Vector2

from draw.vis_util import get_2d_car_model, get_2d_uav_model
from mamp.tools.utils import get_boundaries, l2normsq, normalize, l2norm, path_length
from mamp.configs.config import NEAR_GOAL_THRESHOLD, NEAR_GOAL_THRESHOLD1

simple_plot = False


# img = plt.imread('beijing.jpg', 100)


def plot_car(ax, x, y, yaw, agent_rd, color="red", steer=0.0, truck_color="black"):  # pragma: no cover
    # Vehicle parameters
    if agent_rd > 0.5:
        scale = agent_rd / 2.5
        LENGTH = 4.43 * scale  # [m]
        WIDTH = 2.0 * scale  # [m]
        BACK_TO_WHEEL = 0.9 * scale  # [m]
        WHEEL_LEN = 0.3 * scale  # [m]
        WHEEL_WIDTH = 0.18 * scale  # [m]
        TREAD = 0.7 * scale  # [m]
        WB = 2.62 * scale
    else:
        LENGTH = 0.30  # [m]
        WIDTH = 0.195  # [m]
        BACK_TO_WHEEL = 0.05  # [m]
        WHEEL_LEN = 0.072 / 2  # [m]      轮子长度。真实/2
        WHEEL_WIDTH = 0.03 / 2  # [m]     轮子宽度。真实/2
        TREAD = 0.14 / 2  # [m]
        WB = 0.18
    l_norm = (LENGTH / 2) - BACK_TO_WHEEL
    x = x - l_norm * cos(yaw)
    y = y - l_norm * sin(yaw)
    outline = np.array(
        [[-BACK_TO_WHEEL, (LENGTH - BACK_TO_WHEEL), (LENGTH - BACK_TO_WHEEL),
          -BACK_TO_WHEEL, -BACK_TO_WHEEL],
         [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array(
        [[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH -
          TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[cos(yaw), sin(yaw)], [-sin(yaw), cos(yaw)]])
    Rot2 = np.array([[cos(steer), sin(steer)], [-sin(steer), cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    ax.plot(np.array(outline[0, :]).flatten(), np.array(outline[1, :]).flatten(), color=color)
    ax.plot(np.array(fr_wheel[0, :]).flatten(), np.array(fr_wheel[1, :]).flatten(), color=truck_color)
    ax.plot(np.array(rr_wheel[0, :]).flatten(), np.array(rr_wheel[1, :]).flatten(), color=truck_color)
    ax.plot(np.array(fl_wheel[0, :]).flatten(), np.array(fl_wheel[1, :]).flatten(), color=truck_color)
    ax.plot(np.array(rl_wheel[0, :]).flatten(), np.array(rl_wheel[1, :]).flatten(), color=truck_color)
    ax.arrow(x, y, 0.5 * agent_rd * cos(yaw), 0.5 * agent_rd * sin(yaw),
             fc=color, ec=color, head_width=0.5 * agent_rd, head_length=0.5 * agent_rd)
    ax.plot(x, y, "*", color=color, markersize=6, zorder=5)


def convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame):
    alpha = heading_global_frame
    for point in agent_model:
        x = point[0]
        y = point[1]
        # 进行航向计算
        ll = sqrt(pow(x, 2) + pow(y, 2))
        alpha_model = atan2(y, x)
        alpha_ = alpha + alpha_model - np.pi / 2  # 改加 - np.pi / 2 因为画模型的时候UAV朝向就是正北方向，所以要减去90°
        point[0] = ll * cos(alpha_) + pos_global_frame[0]
        point[1] = ll * sin(alpha_) + pos_global_frame[1]


def draw_agent_2d(ax, pos_global_frame, heading_global_frame, my_agent_model, color='grey', alpha=0.9):
    agent_model = my_agent_model
    convert_to_actual_model_2d(agent_model, pos_global_frame, heading_global_frame)

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY,
             ]

    path = Path(agent_model, codes)
    # 第二步：创建一个patch，路径依然也是通过patch实现的，只不过叫做pathpatch
    col = [0.8, 0.8, 0.8]
    patch = patches.PathPatch(path, fc=color, ec=color, lw=1.0, alpha=0.9)

    ax.add_patch(patch)


def draw_polygon_2d(ax, vertices, fc_color='grey', ec_color='grey', alpha=0.9):
    vertices.append(vertices[0])  # 首尾相连
    codes = []
    for i in range(len(vertices)):
        if i == 0:
            codes.append(Path.MOVETO)
        elif i == len(vertices) - 1:
            codes.append(Path.CLOSEPOLY)
        else:
            codes.append(Path.LINETO)

    path = Path(vertices, codes)
    # 第二步：创建一个patch，路径依然也是通过patch实现的，只不过叫做pathpatch
    col = [0.8, 0.8, 0.8]
    patch = patches.PathPatch(path, fc=fc_color, ec=ec_color, lw=0.8, alpha=alpha)

    ax.add_patch(patch)


def draw_polygon_2d_ins(ax, axins, vertices, color='blue', alpha=0.9):
    vertices.append(vertices[0])  # 首尾相连
    codes = []
    for i in range(len(vertices)):
        if i == 0:
            codes.append(Path.MOVETO)
        elif i == len(vertices) - 1:
            codes.append(Path.CLOSEPOLY)
        else:
            codes.append(Path.LINETO)

    path = Path(vertices, codes)
    # 第二步：创建一个patch，路径依然也是通过patch实现的，只不过叫做pathpatch
    col = [0.8, 0.8, 0.8]

    # rect = patches.Rectangle((min_xe, min_ye), width, height,
    #                          linewidth=2, edgecolor='orange', facecolor='none', zorder=3, alpha=1)
    # ax.add_patch(rect)

    patch = patches.PathPatch(path, fc='grey', ec='grey', lw=1.0, alpha=alpha)
    ax.add_patch(patch)

    # ax.add_patch(plt.PathPatch(path, fc='grey', ec='grey', lw=1.0, alpha=alpha))
    patch1 = patches.PathPatch(path, fc='grey', ec='grey', lw=1.0, alpha=alpha)
    axins.add_patch(patch1)


def draw_traj_2d(ax, tar_and_obs_info, agents_info, agents_traj_list, step_num_list, current_step,
                 task_area, exit_area, plot_target, target_color):
    plt_colors = get_colors()
    for idx, agent_traj in enumerate(agents_traj_list):
        agent_id = agents_info[idx]['id']
        agent_rd = agents_info[idx]['radius']
        agent_goal = agents_info[idx]['goal_pos']
        color_ind = agent_id % len(plt_colors)
        plt_color = plt_colors[color_ind]

        ag_step_num = step_num_list[idx]
        if current_step > ag_step_num - 1:
            plot_step = ag_step_num - 1
        else:
            plot_step = current_step

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        alpha = agent_traj['yaw']

        # 绘制end区域
        if len(exit_area) > 0:
            min_xe, max_xe, min_ye, max_ye = get_boundaries(exit_area)
            width = max_xe - min_xe
            height = max_ye - min_ye
            # rect = patches.Rectangle((min_xe, min_ye), width, height,
            #                          linewidth=2, edgecolor='green', facecolor='none', zorder=3, alpha=1)
            # ax.add_patch(rect)
            # ax.add_patch(plt.Rectangle((min_xe, min_ye), width, height, linewidth=2, edgecolor='green',
            #                            facecolor='none', zorder=3, alpha=1))

        # 绘制task区域
        # min_xt, max_xt, min_yt, max_yt = get_boundaries(task_area)
        # width_t = max_xt - min_xt
        # height_t = max_yt - min_yt
        # rect = patches.Rectangle((min_xt, min_yt), width_t, height_t,
        #                          linewidth=2, edgecolor='blue', facecolor='none', alpha=1)
        # ax.add_patch(rect)
        # ax.add_patch(plt.Rectangle((min_xt, min_yt), width_t, height_t, linewidth=2, edgecolor='blue',
        #                            facecolor='none', alpha=1))

        # # 绘制起始位置
        # ax.add_patch(plt.Circle((pos_x[0], pos_y[0]), radius=agent_rd, fc='none', ec=plt_color, linewidth=1, alpha=1))
        # axins.add_patch(
        #     plt.Circle((pos_x[0], pos_y[0]), radius=agent_rd, fc='none', ec=plt_color, linewidth=1, alpha=1))
        text_offset = agent_rd
        ax.text(pos_x[0] - 1.05 * text_offset, pos_y[0] + 0.8 * text_offset, 'Robot' + str(agent_id),
                color=plt_color)

        # 绘制实线
        plt.plot(pos_x[:plot_step], pos_y[:plot_step], linewidth=1.5, color=plt_color, zorder=3)

        # 绘制经过清除目标的位置
        # for i in range(len(tar_and_obs_info['targets_info'])):
        #     tar_id = tar_and_obs_info['targets_info'][i]['id']
        #     pos = tar_and_obs_info['targets_info'][i]['position']
        #     shape = tar_and_obs_info['targets_info'][i]['shape']
        #     if shape == 'circle':
        #         ob_rd = tar_and_obs_info['targets_info'][i]['feature']
        #     else:
        #         ob_rd1 = tar_and_obs_info['targets_info'][i]['feature']
        #         ob_rd = np.sqrt(ob_rd1[0] ** 2 + ob_rd1[1] ** 2) / 2
        #
        #     # large-scale use NEAR_GOAL_THRESHOLD, real experiment use NEAR_GOAL_THRESHOLD1
        #     if l2norm([pos_x[plot_step], pos_y[plot_step]], pos) < NEAR_GOAL_THRESHOLD1 or plot_target[i]:
        #         plot_target[i] = True
        #         plt.plot(pos[0], pos[1], color=target_color[tar_id], marker='*', markersize=6, alpha=1)
        #         ax.add_patch(plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='none',
        #                                 ec='green', alpha=0.05, linestyle='--', linewidth=0.3))
        #         # axins.plot(pos[0], pos[1], color=target_color[tar_id], marker='*', markersize=3, alpha=1)
        #         # axins.add_patch(plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='none',
        #         #                            ec='green', alpha=0.05, linestyle='--', linewidth=0.3))

        if simple_plot:
            ax.add_patch(plt.Circle((pos_x[plot_step], pos_y[plot_step]), radius=agent_rd, fc=plt_color, ec=plt_color))
        else:
            plot_car(ax, pos_x[plot_step], pos_y[plot_step], alpha[plot_step], agent_rd, color=plt_color)

    for i in range(len(tar_and_obs_info['targets_info'])):
        if tar_and_obs_info['targets_info'][i]['shape'] == 'circle':
            if plot_target[i]:
                pos = tar_and_obs_info['targets_info'][i]['position']
                ob_rd = tar_and_obs_info['targets_info'][i]['feature']
                # ax.text(pos[0], pos[1] + ob_rd, str(i), color='green', alpha=0.66)
            else:
                tar_id = tar_and_obs_info['targets_info'][i]['id']
                pos = tar_and_obs_info['targets_info'][i]['position']
                ob_rd = tar_and_obs_info['targets_info'][i]['radius']
                # ax.add_patch(plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='red', ec='none', alpha=0.02))
                ax.add_patch(plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='none', ec=target_color[tar_id],
                                        alpha=0.8, linewidth=1.0, linestyle='dashed'))
                plt.plot(pos[0], pos[1], color=target_color[tar_id], marker='<', markersize=12, alpha=1.0, zorder=3)
                ax.text(pos[0] - 0.4 * ob_rd, pos[1] + 1.1 * ob_rd, 'task' + str(tar_id), color=target_color[tar_id],
                        alpha=1)

    for i in range(len(tar_and_obs_info['obstacles_info'])):
        pos = tar_and_obs_info['obstacles_info'][i]['position']
        shape = tar_and_obs_info['obstacles_info'][i]['shape']
        if shape == 'circle':
            ob_rd = tar_and_obs_info['obstacles_info'][i]['feature']
            ax.add_patch(plt.Circle((pos[0], pos[1]), radius=ob_rd, fc='grey', ec='grey', alpha=0.9))
        elif shape == 'polygon':
            vertices = tar_and_obs_info['obstacles_info'][i]['vertices']
            draw_polygon_2d(ax, vertices)
        elif shape == 'rect':
            heading = 0.0
            rd = tar_and_obs_info['obstacles_info'][i]['feature']
            agent_rd = rd[0] / 2
            my_model = get_2d_car_model(size=agent_rd)
            draw_agent_2d(ax, pos, heading, my_model, color='grey', alpha=0.9)


def plot_save_one_pic(tar_and_obs_info, agents_info, agents_traj_list, step_num_list, filename,
                      current_step, task_area, exit_area, plot_target, target_color):
    fig = plt.figure(0)
    scale = 1.2
    fig_size = (10 * scale, 7 * scale)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    # ax.set(xlabel='X', ylabel='Y', )
    ax.axis('equal')
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # axins = ax.inset_axes((0.091, 0.05, 0.817, 0.52))
    # axins.axis('equal')
    # axins.set_xticks([])
    # axins.set_yticks([])
    draw_traj_2d(ax, tar_and_obs_info, agents_info, agents_traj_list,
                 step_num_list, current_step, task_area, exit_area, plot_target, target_color)

    fig.savefig(filename)
    if current_step == 0: plt.show()
    if current_step == 2870: plt.show()
    # if current_step == 2620: plt.show()
    # if current_step == 8342: plt.show()
    # if current_step == 15494: plt.show()
    # if current_step == 15626: plt.show()
    # fig.savefig(filename)
    # plt.show()
    plt.close()


def plot_episode(tar_and_obs_info, agents_info, traj_list, step_num_list, plot_save_dir, base_fig_name, last_fig_name,
                 task_area, exit_area, tasks_scheme, show=False):
    current_step = 0
    num_agents = len(step_num_list)
    total_step = max(step_num_list)
    print('num_agents:', num_agents, 'total_step:', total_step)
    assignment_results = tasks_scheme
    plot_target = [False for _ in range(len(tar_and_obs_info['targets_info']))]
    target_color = ['red' for _ in range(len(tar_and_obs_info['targets_info']))]
    plt_colors = get_colors()
    for agent_id in assignment_results:
        color_ind = int(agent_id) % len(plt_colors)
        plt_color = plt_colors[color_ind]
        for tar_id in assignment_results[agent_id]:
            target_color[tar_id] = plt_color

    while current_step < total_step:
        fig_name = base_fig_name + "_{:05}".format(current_step) + '.png'
        filename = plot_save_dir + fig_name
        plot_save_one_pic(tar_and_obs_info, agents_info, traj_list, step_num_list, filename,
                          current_step, task_area, exit_area, plot_target, target_color)
        print(filename)
        current_step += 5
    filename = plot_save_dir + last_fig_name
    plot_save_one_pic(tar_and_obs_info, agents_info, traj_list, step_num_list, filename,
                      total_step, task_area, exit_area, plot_target, target_color)


def get_cmap(N):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    return map_index_to_rgb_color


def get_colors():
    py_colors = np.array(
        [
            [255, 0, 0], [255, 165, 0], [0, 0, 255], [0, 255, 0], [160, 32, 240], [152, 251, 152], (255, 102, 153),
            [132, 112, 255], [0, 255, 255], [255, 69, 0], [148, 0, 211], [255, 192, 203], [255, 99, 71],
            [255, 127, 0], [0, 191, 255], [255, 0, 255],
        ]
    )
    return py_colors / 255


def get_colors_rgb():
    py_colors = np.array(
        [
            [255, 0, 0], [255, 165, 0], [0, 0, 255], [0, 255, 0], [160, 32, 240], [152, 251, 152], [255, 69, 0],
            [255, 99, 71], [132, 112, 255], [0, 255, 255], [255, 69, 0], [148, 0, 211], [255, 192, 203],
            [255, 127, 0], [0, 191, 255], [255, 0, 255],
        ]
    )
    return py_colors


def tap_colors():
    colors1_hex = [
        "#ffcc00", "#ff9933", "#ff6699", "#ff3366", "#cc33ff",
        "#9933ff", "#3399ff", "#3498db", "#00ccff", "#00ff99"
    ]

    # colors1_rgb = [
    #     (244, 208, 63),  # #f4d03f → (244, 208, 63)
    #     (247, 231, 180),  # #f7e7b4 → (247, 231, 180)
    #     (246, 164, 167),  # #f6a4a7 → (246, 164, 167)
    #     (168, 200, 224),  # #a8c8e0 → (168, 200, 224)
    #     (93, 160, 212),  # #5da0d4 → (93, 160, 212)
    #     (44, 62, 80),  # #2c3e50 → (44, 62, 80)
    #     (231, 76, 60),  # #e74c3c → (231, 76, 60)
    #     (52, 152, 219),  # #3498db → (52, 152, 219)
    #     (243, 156, 18),  # #f39c12 → (243, 156, 18)
    #     (142, 68, 173)  # #8e44ad → (142, 68, 173)
    # ]

    colors1_rgb = [
        (255, 204, 0),  # #ffcc00 → (255, 204, 0)
        (255, 153, 51),  # #ff9933 → (255, 153, 51)
        (255, 102, 153),  # #ff6699 → (255, 102, 153)
        (255, 51, 51),  # #ff3366 → (255, 51, 102)
        (204, 51, 255),  # #cc33ff → (204, 51, 255)
        (153, 51, 255),  # #9933ff → (153, 51, 255)
        (51, 153, 255),  # #3399ff → (51, 153, 255)
        (52, 152, 219),  # #3498db → (52, 152, 219)
        (0, 204, 255),  # #00ccff → (0, 204, 255)
        (0, 255, 153)  # #00ff99 → (0, 255, 153)
    ]

    # colors1_rgb = [
    #     (0, 71, 171), (209, 73, 5), (34, 139, 34), (214, 2, 112), (0, 49, 82), (255, 140, 0), (153, 102, 204),
    #     (0, 139, 139), (178, 34, 34), (107, 142, 35)
    # ]
    colors1_rgb = np.array(colors1_rgb) / 255
    return colors1_rgb


def draw_optied_traj_2d(ax, targets, obstacles, agents, agents_traj_list, task_area, exit_area, plot_turn=False,
                        consider_obs=None, dubins=False, dubins_path=None):
    plt_colors1 = get_colors()
    plt_colors = tap_colors()
    target_color = {}
    for idx, agent_traj in enumerate(agents_traj_list):
        agent_goal = agents[idx].goal_
        agent_r = agents[idx].radius_
        color_ind = idx % len(plt_colors)
        plt_color = plt_colors[color_ind]
        for target in agents[idx].visited_targets_ + agents[idx].targets_:
            target_color[target.id] = color_ind

        pos_x = agent_traj['pos_x']
        pos_y = agent_traj['pos_y']
        heading = agents[idx].heading_yaw_
        ag_pos = agents[idx].position_

        # 绘制start区域
        color_start = [255 / 255, 204 / 255, 0 / 255]
        # 绘制task区域
        min_xt, max_xt, min_yt, max_yt = get_boundaries(task_area)
        # max_xt = 5800
        width_t = max_xt - min_xt
        height_t = max_yt - min_yt
        box_colors = rcParams['axes.prop_cycle'].by_key()['color']  # 获取默认颜色循环
        # box_color = box_colors[0 % len(box_colors)]
        color_task = [0 / 255, 102 / 255, 204 / 255]
        rect = patches.Rectangle((min_xt-50, min_yt-50), width_t + 100, height_t + 50,
                                 linewidth=5, edgecolor=color_task, facecolor='none', alpha=1)
        ax.add_patch(rect)

        # 绘制机器人end区域
        if len(exit_area) > 0:
            min_xe, max_xe, min_ye, max_ye = get_boundaries(exit_area)
            width = max_xe - min_xe
            height = max_ye - min_ye
            # box_color1 = box_colors[5 % len(box_colors)]
            color_end = [20 / 255, 25 / 255, 45 / 255]
            rect = patches.Rectangle((min_xe, min_ye-50), width+50, height+50,
                                     linewidth=5, edgecolor=color_end, facecolor='none', alpha=0.2, zorder=3)
            ax.add_patch(rect)

        # # 绘制目标点
        plt.plot(agent_goal.x, agent_goal.y, color=plt_color, marker='<', markersize=16, zorder=5)
        # plt.text(agent_goal.x - 0.25, agent_goal.y + 50, ' ' + str(agents[idx].id_), color=plt_color)

        # # 绘制机器人位置
        plot_car(ax, pos_x[-1], pos_y[-1], heading, agent_r, color=plt_color)
        plt.text(pos_x[-1] + 0.5 * agent_r, pos_y[-1] + 2 * agent_r, str(agents[idx].id_+1), fontsize=12,
                 color='b', zorder=10)

        # # 绘制箭头
        # plt.arrow(ag_pos[0], ag_pos[1], 2 * agents[idx].radius * cos(heading), 2 * agents[idx].radius * sin(heading),
        #           fc=plt_color, ec=plt_color, head_width=1.2 * agent_r, head_length=1.2 * agent_r)

        # 绘制实线
        plt.plot(pos_x[:], pos_y[:], color=plt_color, linewidth=3, alpha=0.5)

        # 绘制箭头路径
        for i in range(len(pos_x) - 1):
            start_x, start_y = pos_x[i], pos_y[i]
            end_x, end_y = pos_x[i + 1], pos_y[i + 1]
            is_target = False
            for tar in targets:
                if l2norm([end_x, end_y], tar.pos_global_frame) < 1e-3 or l2norm([end_x, end_y], [pos_x[-1], pos_y[-1]]) < 1e-3:
                    is_target = True
                    break
            if is_target:
                plt.annotate("", xy=(end_x, end_y), xytext=(start_x, start_y),
                             arrowprops=dict(arrowstyle="->", color=plt_color, lw=3, mutation_scale=30, alpha=0.6, zorder=5))

    ob_colors = rcParams['axes.prop_cycle'].by_key()['color']  # 获取默认颜色循环
    for i in range(len(obstacles)):
        # color = ob_colors[i % len(ob_colors)]
        color = [160 / 255, 160 / 255, 160 / 255]
        pos = obstacles[i].pos_global_frame
        if obstacles[i].shape == 'circle':
            rd = obstacles[i].radius
            ax.add_patch(plt.Circle((pos[0], pos[1]), radius=rd, fc='grey', ec='grey', alpha=0.7))
        elif obstacles[i].shape == 'polygon':
            vertices = copy.deepcopy(obstacles[i].vertices_pos)
            draw_polygon_2d(ax, vertices, fc_color=color, ec_color=color)
            inflated_vertices = copy.deepcopy(agents[0].planner_.inflated_obs[i].vertices_pos)
            draw_polygon_2d(ax, inflated_vertices, fc_color='none', ec_color=color)
        # ax.scatter(pos[0], pos[1], marker='o', color='black', s=10, zorder=3)
    for i in range(len(targets)):
        rd = targets[i].radius
        pos = targets[i].pos_global_frame
        color = plt_colors[target_color[targets[i].id]]
        ax.add_patch(plt.Circle((pos[0], pos[1]), radius=rd, fc='none', ec=color, linewidth=1, linestyle='dashed'))
        ax.scatter(pos[0], pos[1], marker='o', color=color, s=200, alpha=0.7, zorder=3)
        # plt.text(pos[0], pos[1], str(targets[i].id), color=color)

    if plot_turn:
        agent = agents[0]
        norm_vel = rvo_math.normalize(agent.velocity_)
        position_goal = Vector2(agent.now_goal_[0], agent.now_goal_[1])
        combinedRadius = agent.turning_radius_ + agent.radius_
        combinedRadiusSq = combinedRadius * combinedRadius

        normal_left = Vector2(-norm_vel.y, norm_vel.x)
        center_left = agent.position_ + agent.turning_radius_ * normal_left
        s_point1 = agent.position_ - center_left
        start_angle1 = math.degrees(math.atan2(s_point1.y, s_point1.x))
        relativePosition1 = center_left - position_goal
        distSq1 = rvo_math.abs_sq(relativePosition1)

        normal_right = Vector2(norm_vel.y, -norm_vel.x)
        center_right = agent.position_ + agent.turning_radius_ * normal_right
        s_point2 = agent.position_ - center_right
        start_angle2 = math.degrees(math.atan2(s_point2.y, s_point2.x))
        relativePosition2 = center_right - position_goal
        distSq2 = rvo_math.abs_sq(relativePosition2)

        if distSq1 >= combinedRadiusSq:
            leg1 = math.sqrt(distSq1 - combinedRadiusSq)
            left_direction = Vector2(relativePosition1.x * leg1 - relativePosition1.y * combinedRadius,
                                     relativePosition1.x * combinedRadius + relativePosition1.y * leg1) / distSq1
            e_point1 = position_goal + leg1 * left_direction
            end_dir1 = e_point1 - center_left
            end_angle1 = math.degrees(math.atan2(end_dir1.y, end_dir1.x))
            wedge = Wedge((center_left.x, center_left.y), combinedRadius, start_angle1, end_angle1, fill=False,
                          edgecolor='orange', linewidth=1.5, zorder=5)
            ax.add_patch(wedge)
            plt.plot(e_point1.x, e_point1.y, color='orange', marker='o', markersize=5)
            ax.plot([e_point1.x, position_goal.x], [e_point1.y, position_goal.y], color='orange', linewidth=0.5,
                    linestyle='dashed', alpha=0.9)

        if distSq2 >= combinedRadiusSq:
            leg2 = math.sqrt(distSq2 - combinedRadiusSq)
            right_direction = Vector2(relativePosition2.x * leg2 + relativePosition2.y * combinedRadius,
                                      -relativePosition2.x * combinedRadius + relativePosition2.y * leg2) / distSq2
            e_point2 = position_goal + leg2 * right_direction
            end_dir2 = e_point2 - center_right
            end_angle2 = math.degrees(math.atan2(end_dir2.y, end_dir2.x))
            wedge = Wedge((center_right.x, center_right.y), combinedRadius, end_angle2, start_angle2, fill=False,
                          edgecolor='blue', linewidth=1.5, zorder=5)
            ax.add_patch(wedge)
            plt.plot(e_point2.x, e_point2.y, color='blue', marker='o', markersize=5)
            ax.plot([e_point2.x, position_goal.x], [e_point2.y, position_goal.y], color='blue', linewidth=0.5,
                    linestyle='dashed', alpha=0.9)

        turning_rad = agent.turning_radius_
        rs = turning_rad - agent.radius_
        ext_p = agent.position_ + 10 * agent.velocity_
        ax.plot([agent.position_.x, ext_p.x], [agent.position_.y, ext_p.y], color='purple', linewidth=0.5,
                linestyle='dashed', alpha=0.9)
        ax.add_patch(plt.Circle((agent.position_.x, agent.position_.y), radius=agent.radius_, fc='none', ec='purple',
                                linewidth=1, linestyle='dashed', zorder=5))
        plt.plot(center_left.x, center_left.y, color='red', marker='o', markersize=5)
        plt.plot(center_right.x, center_right.y, color='blue', marker='o', markersize=5)
        ax.add_patch(plt.Circle((center_left.x, center_left.y), radius=combinedRadius, fc='none', ec='red', linewidth=1,
                                linestyle='dashed', alpha=0.2, zorder=5))
        ax.add_patch(plt.Circle((center_right.x, center_right.y), radius=combinedRadius, fc='none', ec='red',
                                linewidth=1, linestyle='dashed', alpha=0.2, zorder=5))
        ax.add_patch(plt.Circle((center_left.x, center_left.y), radius=turning_rad, fc='none', ec='red', linewidth=0.5,
                                linestyle='dashed', alpha=0.8, zorder=5))
        ax.add_patch(plt.Circle((center_left.x, center_left.y), radius=rs, fc='none', ec='red', linewidth=0.5,
                                linestyle='dashed', alpha=0.8, zorder=5))
        ax.add_patch(plt.Circle((center_right.x, center_right.y), radius=turning_rad, fc='none', ec='red',
                                linewidth=0.5, linestyle='dashed', alpha=0.8, zorder=5))
        ax.add_patch(plt.Circle((center_right.x, center_right.y), radius=rs, fc='none', ec='red',
                                linewidth=0.5, linestyle='dashed', alpha=0.8, zorder=5))
        ax.quiver(agent.position_.x, agent.position_.y, agent.velocity_.x, agent.velocity_.y,
                  angles='xy', scale_units='xy', scale=1, color='purple', zorder=5)
        for i in range(len(consider_obs)):
            obstacle1, obstacle2 = consider_obs[i]
            if obstacle1 == obstacle2:
                obstacle2 = obstacle1.next_
            p1, p2 = obstacle1.point_, obstacle2.point_
            ax.plot([p1.x, p2.x], [p1.y, p2.y], color='black', linewidth=3.5, zorder=6)

    if dubins:
        if dubins_path is not None:
            px = [p[0] for p in dubins_path]
            py = [p[1] for p in dubins_path]
            ax.plot(px, py, color=plt_colors[agents[0].id_], marker='o', linewidth=1.5, linestyle='dashed', alpha=0.9)


def draw_sector_and_line(ax, center, r, start_angle, end_angle, p1, p2):
    """
    绘制扇形和线段的示意图
    """
    cx, cy = center
    # 绘制扇形
    wedge = Wedge((cx, cy), r, start_angle, end_angle, fill=False, edgecolor='blue')
    ax.add_patch(wedge)


def plt_visulazation(path_smooth, robots, targets, obstacles, task_area, exit_area, plot_turn=False,
                     consider_obs=None, dubins=False, dubins_path=None):
    trajs = []
    path_length_list = []
    for path in path_smooth:
        traj = {'pos_x': [], 'pos_y': [], 'spd': []}
        path_length_list.append(path_length(path))
        for ii in range(len(path)):
            traj['pos_x'].append(path[ii][0])
            traj['pos_y'].append(path[ii][1])
        trajs.append(traj)

    fig = plt.figure(0)
    fig_size = (15 * 1, 12 * 1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    ax.set(xlabel='X',
           ylabel='Y',
           )
    ax.axis('equal')
    draw_optied_traj_2d(ax, targets, obstacles, robots, trajs, task_area, exit_area, plot_turn=plot_turn,
                        consider_obs=consider_obs, dubins=dubins, dubins_path=dubins_path)
    # plt.axis('off')
    # plt.grid(True)
    # plt.xticks(np.arange(-0, 7.0, 0.6))  # x轴每隔0.6个单位显示一个刻度
    # plt.yticks(np.arange(-0, 7.0, 0.6))  # y轴每隔0.6个单位显示一个刻度
    from matplotlib import font_manager
    plt.rcParams['pdf.fonttype'] = 42  # 确保PDF使用Type 1字体
    plt.rcParams['ps.fonttype'] = 42  # 确保PostScript使用Type 1字体
    # plt.rcParams['font.family'] = 'serif'  # 设置字体家族
    # plt.rcParams['font.serif'] = ['Times New Roman']  # 设置字体为Times New Roman
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'  # 确保路径正确
    prop = font_manager.FontProperties(fname=font_path)
    path_dist = round(max(path_length_list)/1000, 2)
    plt.text(1010, 4120, f'Maximum travel distance among robots: {path_dist} km', fontsize=25, color='black', fontproperties=prop)
    plt.axis("equal")
    plt.show()


def plot_half_planes(half_planes, agent, new_velocity):
    """
    :param half_planes: 半平面的线性约束
    :param agent:
    :param new_velocity:
    """
    radius, pref_velocity = agent.max_speed_, agent.pref_velocity_

    fig = plt.figure(0)
    fig_size = (10 * 1, 8 * 1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-5.0 * radius, 5.0 * radius])
    ax.set_ylim([-5.0 * radius, 5.0 * radius])
    ax.set_aspect('equal')

    # 绘制每个半平面
    # print(pref_velocity.x, pref_velocity.y)
    for i in range(len(half_planes)):
        point = np.array([half_planes[i].point.x, half_planes[i].point.y])
        direction = np.array([half_planes[i].direction.x, half_planes[i].direction.y])
        normal = np.array([-half_planes[i].direction.y, half_planes[i].direction.x])
        x, y = point
        dx, dy = direction

        # 生成边界线
        p1 = point - 2 * radius * direction
        p2 = point + 2 * radius * direction
        line_x = [p1[0], point[0], p2[0]]
        line_y = [p1[1], point[1], p2[1]]

        # 生成半平面，长方形代替
        p3 = p1 + 0.1 * radius * normal
        p4 = p2 + 0.1 * radius * normal
        vertices = [p1, p2, p4, p3]

        # 绘制半平面
        draw_polygon_2d(ax, vertices, fc_color='grey', ec_color='grey', alpha=0.6)

        # 绘制边界线
        ax.plot(line_x, line_y, color='red', linewidth=1.5, linestyle='dashed')
        # 绘制orca_line的起点
        ax.plot(x, y, color='red', marker='o')

        # 绘制方向向量
        rad = 0.5 * radius
        h = 0.1 * radius
        plt.arrow(x, y, rad * dx, rad * dy, fc='DarkOrange', ec='DarkOrange', head_width=h, head_length=h, zorder=3)
        plt.text(x, y, str(i), color='blue', zorder=5)

    ax.add_patch(plt.Circle((0.0, 0.0), radius=radius, fc='none', ec='red', linewidth=1.5, linestyle='dashed'))
    # 绘制速度
    labels = ['pref_velocity', 'new_velocity', 'velocity']
    velocity = agent.velocity_
    ax.quiver(0, 0, pref_velocity.x, pref_velocity.y, angles='xy', scale_units='xy', scale=1, color='green',
              label=labels[0], zorder=3)
    ax.quiver(0, 0, new_velocity.x, new_velocity.y, angles='xy', scale_units='xy', scale=1, color='red', zorder=5,
              label=labels[1])
    ax.quiver(0, 0, velocity.x, velocity.y, angles='xy', scale_units='xy', scale=1, color='purple', zorder=3,
              label=labels[2])

    # 绘制四象限
    ax.plot([-1, 1], [0, 0], color='black', linewidth=1.5)
    ax.plot([0, 0], [-1, 1], color='black', linewidth=1.5)

    # 显示绘图
    plt.xlabel('X')
    plt.ylabel('Y')
    font2 = {'weight': 'normal', 'size': 16}
    plt.legend(loc='upper left', prop=font2)
    plt.title('ORCA Half Planes')
    plt.show()


def plot_half_planes1(new_velocity, velocity, new_velocity1):
    radius, pref_velocity = 0.12, new_velocity

    fig = plt.figure(0)
    fig_size = (10 * 1, 8 * 1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim([-5.0 * radius, 5.0 * radius])
    ax.set_ylim([-5.0 * radius, 5.0 * radius])
    ax.set_aspect('equal')
    labels = ['pref_velocity', 'new_velocity', 'velocity']

    ax.add_patch(plt.Circle((0.0, 0.0), radius=radius, fc='none', ec='red', linewidth=1.5, linestyle='dashed'))
    # 绘制最优速度
    ax.quiver(0, 0, pref_velocity.x, pref_velocity.y, angles='xy', scale_units='xy', scale=1, color='green',
              label=labels[0], zorder=5)
    ax.quiver(0, 0, new_velocity1.x, new_velocity1.y, angles='xy', scale_units='xy', scale=1, color='blue', zorder=3,
              label=labels[1])
    ax.quiver(0, 0, velocity.x, velocity.y, angles='xy', scale_units='xy', scale=1, color='red', zorder=3,
              label=labels[2])

    # 绘制四象限
    ax.plot([-1, 1], [0, 0], color='black', linewidth=1.5)
    ax.plot([0, 0], [-1, 1], color='black', linewidth=1.5)

    # 显示绘图
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('ORCA Half Planes')
    font2 = {'weight': 'normal', 'size': 16}
    plt.legend(loc='upper left', prop=font2)
    plt.show()


def plot_vo(agent, other, invTimeHorizon, half_planes, idx, u, w):
    """
    绘制 Velocity Obstacle
    """
    fig = plt.figure(0)
    fig_size = (10 * 1, 8 * 1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    scale = 18
    ax.set_xlim([-scale * agent.radius_, scale * agent.radius_])
    ax.set_ylim([-scale * agent.radius_, scale * agent.radius_])
    ax.set_aspect('equal')

    agent_pos = np.array([agent.position_.x, agent.position_.y])
    offset = agent_pos
    agent_pos = agent_pos - offset
    agent_vel = np.array([agent.velocity_.x, agent.velocity_.y])
    agent_pref_vel = np.array([agent.pref_velocity_.x, agent.pref_velocity_.y])
    agent_rad = agent.radius_ + agent.planner_.inflation

    other_pos = np.array([other.position_.x, other.position_.y]) - offset
    other_vel = np.array([other.velocity_.x, other.velocity_.y])
    other_rad = other.radius_ + agent.planner_.inflation

    combinedRadius = agent_rad + other_rad
    relativePosition = other_pos - agent_pos
    relative_vel = agent_vel - other_vel
    relative_opt_vel = agent_pref_vel - other_vel
    inv_pos = invTimeHorizon * (other_pos - agent_pos)
    inv_combinedRadius = invTimeHorizon * combinedRadius

    unit_relativePosition = normalize(relativePosition)
    n_left = np.array([-unit_relativePosition[1], unit_relativePosition[0]])
    n_right = np.array([unit_relativePosition[1], -unit_relativePosition[0]])
    small_left1 = inv_pos + combinedRadius * n_left
    small_right1 = inv_pos + combinedRadius * n_right
    large_left1 = other_pos + combinedRadius * n_left
    large_right1 = other_pos + combinedRadius * n_right

    distSq = l2normsq(agent_pos, other_pos)
    leg = np.sqrt(distSq - combinedRadius ** 2)
    left_direction = normalize(np.array([relativePosition[0] * leg - relativePosition[1] * combinedRadius,
                                         relativePosition[0] * combinedRadius + relativePosition[1] * leg]))
    right_direction = normalize(np.array([relativePosition[0] * leg + relativePosition[1] * combinedRadius,
                                          -relativePosition[0] * combinedRadius + relativePosition[1] * leg]))
    large_left = agent_pos + leg * left_direction
    large_right = agent_pos + leg * right_direction
    small_left = agent_pos + invTimeHorizon * leg * left_direction
    small_right = agent_pos + invTimeHorizon * leg * right_direction
    small_left2 = inv_pos + combinedRadius * normalize(small_left - inv_pos)
    small_right2 = inv_pos + combinedRadius * normalize(small_right - inv_pos)

    # 绘制VO的边界
    ax.plot([agent_pos[0], large_left[0]], [agent_pos[1], large_left[1]], color='black', linewidth=0.5,
            linestyle='dashed', alpha=0.2)
    ax.plot([agent_pos[0], large_right[0]], [agent_pos[1], large_right[1]], color='black', linewidth=0.5,
            linestyle='dashed', alpha=0.2)
    ax.plot([small_left[0], large_left[0]], [small_left[1], large_left[1]], color='red', linewidth=1.0,
            linestyle='dashed')
    ax.plot([small_right[0], large_right[0]], [small_right[1], large_right[1]], color='red', linewidth=1.0,
            linestyle='dashed')
    # ax.quiver(small_left[0], small_left[1], large_left[0], large_left[1], angles='xy', scale_units='xy', scale=1,
    #           color='red')
    # ax.quiver(small_right[0], small_right[1], large_right[0], large_right[1], angles='xy', scale_units='xy', scale=1,
    #           color='red')

    # 绘制相切大圆的垂直辅助线
    ax.plot([other_pos[0], large_left[0]], [other_pos[1], large_left[1]], color='red', linewidth=0.5,
            linestyle='dashed')
    ax.plot([other_pos[0], large_right[0]], [other_pos[1], large_right[1]], color='red', linewidth=0.5,
            linestyle='dashed')
    # 绘制相切小圆的垂直辅助线
    ax.plot([inv_pos[0], small_left2[0]], [inv_pos[1], small_left2[1]], color='red', linewidth=0.5,
            linestyle='dashed')
    ax.plot([inv_pos[0], small_right2[0]], [inv_pos[1], small_right2[1]], color='red', linewidth=0.5,
            linestyle='dashed')

    # 绘制大圆的垂直相对位置辅助线
    ax.plot([other_pos[0], large_left1[0]], [other_pos[1], large_left1[1]], color='black', linewidth=0.5,
            linestyle='dashed')
    ax.plot([other_pos[0], large_right1[0]], [other_pos[1], large_right1[1]], color='black', linewidth=0.5,
            linestyle='dashed')
    # 绘制小圆的垂直相对位置辅助线
    ax.plot([inv_pos[0], small_left1[0]], [inv_pos[1], small_left1[1]], color='black', linewidth=0.5,
            linestyle='dashed')
    ax.plot([inv_pos[0], small_right1[0]], [inv_pos[1], small_right1[1]], color='black', linewidth=0.5,
            linestyle='dashed')

    # 绘制agent以及other
    ax.plot([agent_pos[0], other_pos[0]], [agent_pos[1], other_pos[1]], color='black', linewidth=1.0,
            linestyle='dashed')
    ax.plot(agent_pos[0], agent_pos[1], color='red', marker='o')
    ax.add_patch(plt.Circle((agent_pos[0], agent_pos[1]), radius=agent_rad, fc='none', ec='red',
                            linewidth=1.0, linestyle='dashed'))
    ax.add_patch(plt.Circle((other_pos[0], other_pos[1]), radius=other_rad, fc='none', ec='blue',
                            linewidth=1.5, linestyle='dashed'))
    ax.plot(other_pos[0], other_pos[1], color='blue', marker='o')
    ax.add_patch(plt.Circle((other_pos[0], other_pos[1]), radius=combinedRadius, fc='none', ec='red',
                            linewidth=1.0, linestyle='dashed'))
    ax.plot(inv_pos[0], inv_pos[1], color='red', marker='o')
    ax.add_patch(plt.Circle((inv_pos[0], inv_pos[1]), radius=inv_combinedRadius, fc='none', ec='red',
                            linewidth=1.0, linestyle='dashed'))

    plt.text(0, 0, 'agent-'+str(agent.id_), color='red')
    plt.text(other_pos[0], other_pos[1], 'other-' + str(other.id_), color='blue')

    labels = ['agent_pref_velocity', 'agent_velocity', 'other_pref_velocity', 'other_velocity', 'relative_velocity']
    # 绘制agent速度
    ax.quiver(0, 0, agent_vel[0], agent_vel[1], angles='xy', scale_units='xy', scale=1, color='red', label=labels[1])
    # 绘制other速度
    h = 0.1 * agent.max_speed_
    plt.arrow(other_pos[0], other_pos[1], other_vel[0], other_vel[1], fc='blue', ec='blue',
              head_width=h, head_length=h, zorder=3)
    ax.quiver(0, 0, other_vel[0], other_vel[1], angles='xy', scale_units='xy', scale=1, color='blue', label=labels[3])

    # 绘制agent相对于other的速度
    ax.quiver(0, 0, relative_vel[0], relative_vel[1], angles='xy', scale_units='xy', scale=1, color='orange', zorder=5,
              label=labels[4])

    # 绘制agent最优速度
    ax.quiver(0, 0, agent_pref_vel[0], agent_pref_vel[1], angles='xy', scale_units='xy', scale=1, color='green',
              label=labels[0])
    # 绘制other最优速度
    ax.quiver(0, 0, other.pref_velocity_.x, other.pref_velocity_.y, angles='xy', scale_units='xy', scale=1,
              color='DarkOrange', label=labels[2])

    # 绘制四象限
    ax.plot([-scale * agent.radius_, scale * agent.radius_], [0, 0], color='black', linewidth=1.5)
    ax.plot([0, 0], [-scale * agent.radius_, scale * agent.radius_], color='black', linewidth=1.5)

    radius = agent.max_speed_
    for i in range(len(half_planes)):
        point = np.array([half_planes[i].point.x, half_planes[i].point.y])
        direction = np.array([half_planes[i].direction.x, half_planes[i].direction.y])
        normal = np.array([-half_planes[i].direction.y, half_planes[i].direction.x])
        x, y = point
        dx, dy = direction

        # 生成边界线
        p1 = point - 2 * radius * direction
        p2 = point + 2 * radius * direction
        line_x = [p1[0], point[0], p2[0]]
        line_y = [p1[1], point[1], p2[1]]

        # 生成半平面，长方形代替
        p3 = p1 + 0.1 * radius * normal
        p4 = p2 + 0.1 * radius * normal
        vertices = [p1, p2, p4, p3]

        # 绘制半平面
        draw_polygon_2d(ax, vertices, fc_color='grey', ec_color='grey', alpha=0.6)

        # 绘制边界线
        ax.plot(line_x, line_y, color='DodgerBlue', linewidth=1.5, linestyle='dashed')
        ax.add_patch(
            plt.Circle((0.0, 0.0), radius=radius, fc='none', ec='DodgerBlue', linewidth=1.5, linestyle='dashed'))
        # 绘制orca_line的起点
        ax.plot(x, y, color='red', marker='o')

        ax.quiver(relative_vel[0], relative_vel[1], u.x, u.y, angles='xy', scale_units='xy', scale=1, color='purple',
                  zorder=5)

        u1 = 0.5 * u
        ax.quiver(agent_vel[0], agent_vel[1], u1.x, u1.y, angles='xy', scale_units='xy', scale=1, color='purple')
        ax.quiver(inv_pos[0], inv_pos[1], w.x, w.y, angles='xy', scale_units='xy', scale=1, color='black')

        # 绘制方向向量
        rad = 0.5 * radius
        plt.arrow(x, y, rad * dx, rad * dy, fc='DodgerBlue', ec='DodgerBlue', head_width=h, head_length=h)
        plt.text(x, y, str(idx), fontsize=8, color='blue')

    # 显示绘图
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velocity Obstacle')
    plt.legend()
    plt.show()


def plot_obs_vo(agent, obstacle1, obstacle2, invTimeHorizonObst, half_planes, idx):
    """
    绘制 Velocity Obstacle
    """
    fig = plt.figure(0)
    fig_size = (10 * 1, 8 * 1)
    fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    scale = 18
    ax.set_xlim([-scale * agent.radius_, scale * agent.radius_])
    ax.set_ylim([-scale * agent.radius_, scale * agent.radius_])
    ax.set_aspect('equal')

    agent_vel = np.array([agent.velocity_.x, agent.velocity_.y])
    agent_pref_vel = np.array([agent.pref_velocity_.x, agent.pref_velocity_.y])
    offset = np.array([agent.position_.x, agent.position_.y])
    agent_rad = agent.radius_ + 2 * agent.planner_.inflation

    relativePosition1 = obstacle1.point_ - agent.position_
    relativePosition2 = obstacle2.point_ - agent.position_

    distSq1 = rvo_math.abs_sq(relativePosition1)
    distSq2 = rvo_math.abs_sq(relativePosition2)

    radiusSq = rvo_math.square(agent_rad)

    obstacleVector = obstacle2.point_ - obstacle1.point_
    s = (-relativePosition1 @ obstacleVector) / rvo_math.abs_sq(obstacleVector)
    distSqLine = rvo_math.abs_sq(-relativePosition1 - s * obstacleVector)

    # Usual situation.
    leg1 = sqrt(distSq1 - radiusSq)
    leftLegDirection = Vector2(relativePosition1.x * leg1 - relativePosition1.y * agent_rad,
                               relativePosition1.x * agent_rad + relativePosition1.y * leg1) / distSq1

    leg2 = sqrt(distSq2 - radiusSq)
    rightLegDirection = Vector2(relativePosition2.x * leg2 + relativePosition2.y * agent_rad,
                                -relativePosition2.x * agent_rad + relativePosition2.y * leg2) / distSq2

    inv_pos1 = invTimeHorizonObst * relativePosition1
    inv_pos2 = invTimeHorizonObst * relativePosition2
    inv_radius = invTimeHorizonObst * agent_rad

    agent_pos = np.array([agent.position_.x, agent.position_.y]) - offset
    ob_pos1 = np.array([obstacle1.point_.x, obstacle1.point_.y]) - offset
    ob_pos2 = np.array([obstacle2.point_.x, obstacle2.point_.y]) - offset

    left_direction = np.array([leftLegDirection.x, leftLegDirection.y])
    right_direction = np.array([rightLegDirection.x, rightLegDirection.y])

    large_left = agent_pos + leg1 * left_direction
    large_right = agent_pos + leg2 * right_direction
    small_left = agent_pos + leg1 * invTimeHorizonObst * left_direction
    small_right = agent_pos + leg2 * invTimeHorizonObst * right_direction
    large_left_extend = agent_pos + 5 * leg1 * left_direction
    large_right_extend = agent_pos + 5 * leg2 * right_direction
    large_vertical_left = ob_pos1 + 4 * agent_rad * normalize(large_left - ob_pos1)
    large_vertical_right = ob_pos2 + 4 * agent_rad * normalize(large_right - ob_pos2)
    inv_p1 = np.array([inv_pos1.x, inv_pos1.y])
    inv_p2 = np.array([inv_pos2.x, inv_pos2.y])
    inv_vertical_left = inv_p1 + 4 * agent_rad * normalize(small_left - inv_p1)
    inv_vertical_right = inv_p2 + 4 * agent_rad * normalize(small_right - inv_p2)

    p1_p2 = np.array([obstacle1.direction_.x, obstacle1.direction_.y])
    n_left = np.array([-p1_p2[1], p1_p2[0]])
    n_right = np.array([p1_p2[1], -p1_p2[0]])
    p1_up = inv_p1 + 10 * agent_rad * n_left
    p1_down = inv_p1 + 10 * agent_rad * n_right
    p2_up = inv_p2 + 10 * agent_rad * n_left
    p2_down = inv_p2 + 10 * agent_rad * n_right
    p1_up1 = ob_pos1 + 1 * agent_rad * n_left
    p1_down1 = ob_pos1 + 1 * agent_rad * n_right
    p2_up1 = ob_pos2 + 1 * agent_rad * n_left
    p2_down1 = ob_pos2 + 1 * agent_rad * n_right
    p1_down2 = inv_p1 + agent_rad * invTimeHorizonObst * n_right
    p2_down2 = inv_p2 + agent_rad * invTimeHorizonObst * n_right
    h = 0.1 * agent.max_speed_
    scale = 5

    # 绘制VO的边界
    ax.plot([agent_pos[0], large_left_extend[0]], [agent_pos[1], large_left_extend[1]], color='red', linewidth=1.5,
            linestyle='dashed', alpha=1.0)
    ax.plot([agent_pos[0], large_right_extend[0]], [agent_pos[1], large_right_extend[1]], color='red', linewidth=1.5,
            linestyle='dashed', alpha=1.0)
    plt.arrow(small_left[0], small_left[1], large_left[0], large_left[1], fc='red', ec='red',
              head_width=scale * h, head_length=scale * h, linestyle='dashed')
    plt.arrow(small_right[0], small_right[1], large_right[0], large_right[1], fc='red', ec='red',
              head_width=scale * h, head_length=scale * h, linestyle='dashed')
    ax.plot([p1_up1[0], p2_up1[0]], [p1_up1[1], p2_up1[1]], color='red', linewidth=1.5, linestyle='dashed')
    ax.plot([p1_down1[0], p2_down1[0]], [p1_down1[1], p2_down1[1]], color='red', linewidth=1.5, linestyle='dashed')
    ax.plot([p1_down2[0], p2_down2[0]], [p1_down2[1], p2_down2[1]], color='red', linewidth=6.0)
    # 绘制相切大圆的垂直辅助线
    ax.plot([ob_pos1[0], large_vertical_left[0]], [ob_pos1[1], large_vertical_left[1]], color='black', linewidth=0.5,
            linestyle='dashed')
    ax.plot([ob_pos2[0], large_vertical_right[0]], [ob_pos2[1], large_vertical_right[1]], color='black', linewidth=0.5,
            linestyle='dashed')
    # 绘制相切小圆的垂直辅助线
    ax.plot([inv_pos1.x, inv_vertical_left[0]], [inv_pos1.y, inv_vertical_left[1]], color='black', linewidth=0.5,
            linestyle='dashed')
    ax.plot([inv_pos2.x, inv_vertical_right[0]], [inv_pos2.y, inv_vertical_right[1]], color='black', linewidth=0.5,
            linestyle='dashed')
    # 绘制垂直线段的辅助线
    ax.plot([p1_down[0], p1_up[0]], [p1_down[1], p1_up[1]], color='black', linewidth=0.5, linestyle='dashed')
    ax.plot([p2_down[0], p2_up[0]], [p2_down[1], p2_up[1]], color='black', linewidth=0.5, linestyle='dashed')

    # 绘制障碍物顶点的圆
    ax.plot([ob_pos1[0], ob_pos2[0]], [ob_pos1[1], ob_pos2[1]], color='black', linewidth=1.5)
    ax.plot(ob_pos1[0], ob_pos1[1], color='black', marker='o')
    ax.plot(ob_pos2[0], ob_pos2[1], color='black', marker='o')
    ax.add_patch(plt.Circle((ob_pos1[0], ob_pos1[1]), radius=agent_rad, fc='none', ec='red',
                            linewidth=1.5, linestyle='dashed'))
    ax.add_patch(plt.Circle((ob_pos2[0], ob_pos2[1]), radius=agent_rad, fc='none', ec='red',
                            linewidth=1.5, linestyle='dashed'))

    ax.plot([inv_pos1.x, inv_pos2.x], [inv_pos1.y, inv_pos2.y], color='black', linewidth=1.5)
    ax.plot(inv_pos1.x, inv_pos1.y, color='black', marker='o')
    ax.plot(inv_pos2.x, inv_pos2.y, color='black', marker='o')
    ax.add_patch(plt.Circle((inv_pos1.x, inv_pos1.y), radius=inv_radius, fc='none', ec='red',
                            linewidth=1.5, linestyle='dashed'))
    ax.add_patch(plt.Circle((inv_pos2.x, inv_pos2.y), radius=inv_radius, fc='none', ec='red',
                            linewidth=1.5, linestyle='dashed'))

    # 绘制agent速度和位置
    ax.quiver(0, 0, agent_vel[0], agent_vel[1], angles='xy', scale_units='xy', scale=1, color='red', zorder=5)
    ax.plot(agent_pos[0], agent_pos[1], color='red', marker='o')

    # 绘制agent最优速度
    ax.quiver(0, 0, agent_pref_vel[0], agent_pref_vel[1], angles='xy', scale_units='xy', scale=1, color='green')

    # 绘制四象限
    ax.plot([-scale * agent.radius_, scale * agent.radius_], [0, 0], color='black', linewidth=1.5)
    ax.plot([0, 0], [-scale * agent.radius_, scale * agent.radius_], color='black', linewidth=1.5)

    radius = agent.max_speed_
    for i in range(len(half_planes)):
        point = np.array([half_planes[i].point.x, half_planes[i].point.y])
        direction = np.array([half_planes[i].direction.x, half_planes[i].direction.y])
        normal = np.array([-half_planes[i].direction.y, half_planes[i].direction.x])
        x, y = point
        dx, dy = direction

        # 生成边界线
        p1 = point - 2 * radius * direction
        p2 = point + 2 * radius * direction
        line_x = [p1[0], point[0], p2[0]]
        line_y = [p1[1], point[1], p2[1]]

        # 生成半平面，长方形代替
        p3 = p1 + 0.1 * radius * normal
        p4 = p2 + 0.1 * radius * normal
        vertices = [p1, p2, p4, p3]

        # 绘制半平面
        draw_polygon_2d(ax, vertices, fc_color='grey', ec_color='grey', alpha=0.6)

        # 绘制边界线
        ax.plot(line_x, line_y, color='DodgerBlue', linewidth=1.5, linestyle='dashed')
        ax.add_patch(
            plt.Circle((0.0, 0.0), radius=radius, fc='none', ec='DodgerBlue', linewidth=1.5, linestyle='dashed'))
        # 绘制orca_line的起点
        ax.plot(x, y, color='red', marker='o')

        # u = 0.5 * u
        # ax.quiver(agent_vel[0], agent_vel[1], u.x, u.y, angles='xy', scale_units='xy', scale=1, color='purple')

        # 绘制方向向量
        rad = 0.5 * radius
        plt.arrow(x, y, rad * dx, rad * dy, fc='DodgerBlue', ec='DodgerBlue', head_width=h, head_length=h)
        plt.text(x, y, str(idx), fontsize=8, color='blue')

    # 显示绘图
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Velocity Obstacle')
    plt.show()
