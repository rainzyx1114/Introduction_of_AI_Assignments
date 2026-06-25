from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
def iswall(x, y, walls):
    b_x = np.floor(x)
    b_y = np.floor(y)
    for dx in range(2):
        for dy in range(2):
            t_x = b_x + dx
            t_y = b_y + dy
            if (t_x, t_y) in walls and abs(t_x - x) <= 0.5 and abs(t_y - y) <= 0.5:
                return True
    return False
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    for _ in range(N):
        all_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    min_x, max_x = np.min(walls[:, 0]), np.max(walls[:, 0])
    min_y, max_y = np.min(walls[:, 1]), np.max(walls[:, 1])
    cnt = 0
    walls_set = set(map(tuple, walls))
    while cnt < N:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        t = np.random.uniform(-np.pi, np.pi)
        if not iswall(x, y, walls_set):
            all_particles[cnt] = Particle(x, y, t, 1 / N)
            cnt += 1
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    k = 0.4
    error = np.linalg.norm(gt - estimated)
    # print(error.max(), error.min())
    weight = np.exp(-error * k)
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    for _ in range(len(particles)):
        resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    N = len(particles)
    pre_sum = [0] * (N + 1)
    for i in range(0, N):
        pre_sum[i + 1] = pre_sum[i] + particles[i].get_weight()
    walls_set = set(map(tuple, walls))
    cnt = 0
    while cnt < N:
        tmp = np.random.uniform(0, pre_sum[N] - 1e-6)
        idx = np.searchsorted(pre_sum, tmp, side='right') - 1
        p = particles[idx]
        new_p = Particle(p.position[0] + np.random.normal(0, 0.1), p.position[1] + np.random.normal(0, 0.1), p.theta + np.random.normal(0, 0.1), 1 / N)
        if not iswall(new_p.position[0], new_p.position[1], walls_set):
            resampled_particles[cnt] = new_p
            cnt += 1
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta = (p.theta + dtheta) #+ np.random.normal(0, 0.01))
    p.theta %= 2 * np.pi
    d = traveled_distance #+ np.random.normal(0, 0.01)
    p.position = p.position + np.array([d * np.cos(p.theta), d * np.sin(p.theta)])
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    # total_weight = sum(p.weight for p in particles)
    # x = sum(p.position[0] * p.weight for p in particles) / total_weight
    # y = sum(p.position[1] * p.weight for p in particles) / total_weight
    # t = sum(p.theta * p.weight for p in particles) / total_weight
    # t %= 2 * np.pi
    # final_result = Particle(x, y, t)
    particles.sort(key=Particle.get_weight, reverse=True)
    final_result = particles[0]
    ### 你的代码 ###
    return final_result