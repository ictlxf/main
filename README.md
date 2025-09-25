# 逆强化学习(Inverse Reinforcement Learning, IRL) 和 RL

## 目录
1. [数据集准备](#1-数据集准备dataloader)
2. [环境配置](#2-conda-及其-env-的配置): [gym](#1-open-gym)
3. [RL (强化学习)](#3-rl的复现和再生产)
4. [IRL (逆强化学习)](#4-irl的复现和再生产)




## 1. 数据集准备(Dataloader)

D4RL数据集，专为RL和IRL准备，其中包括[D4RL-V1](Data%2FD4RL-V1)，[D4RL-v2](Data%2FD4RL-v2)，以及六个D4RL的子集数据集：
1. [d4rl_adroit_relocate_cloned_v2](Data%2Fd4rl_adroit_relocate_cloned_v2)
2. [d4rl_adroit_relocate_cloned_v2_done](Data%2Fd4rl_adroit_relocate_cloned_v2_done)
3. [d4rl_adroit_relocate_expert_v2](Data%2Fd4rl_adroit_relocate_expert_v2)
4. [d4rl_adroit_relocate_expert_v2_done](Data%2Fd4rl_adroit_relocate_expert_v2_done)
5. [d4rl_adroit_relocate_human_v2](Data%2Fd4rl_adroit_relocate_human_v2)
6. [d4rl_adroit_relocate_human_v2_done](Data%2Fd4rl_adroit_relocate_human_v2_done)


## 2. conda 及其 env 的配置



Anaconda可以作为python环境管理的基本工具，能够方便快捷地创建不同版本的python环境，并且每个环境相互隔离，不会干扰产生环境变量干扰的问题。

查看 conda 环境列表： `conda env list`， 查看安装包：`conda list`.
创建 conda 环境： `conda create -n myenv python=3.10` 其中 **myenv** 是环境名字，**python** 版本为3.10.
激活 conda 环境： `conda activate myenv`, 安装指定包: `conda install numpy`。
安装所有程序包`pip install -r requirements.txt`。
导出conda 环境 `conda env export > environment.yml`
导入coda 环境`conda env create -f environment.yml`，其中**environment.yml**为文件名。
导出pip安装的包`pip freeze > requirements.txt`
导入pip安装的包`pip install -r requirements.txt`

创建基于python3.6 名为env_name的环境
`conda create --name env_name python=3.6`

激活(使用)环境
`conda activate env_name`

激活后安装各种包(pip是安装在整个计算机里，conda是安装当前虚拟环境里)
`conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`

重命名(本质上是克隆环境然后删除原环境)
`conda create --name new_env_name --clone old_evn_name`

删除环境
`conda remove -n env_name --all`

删除环境中的某个包
`conda remove --name $env_name  $package_name`

查看安装了哪些包
`conda list`

查看有哪些环境
`conda env list`

更新conda
`conda update conda`

激活需要导出的环境
`conda activate env_name`

生成相关yaml文件，文件会生成在C盘用户目录里
`conda env export > your_env.yaml`

在新电脑上根据yaml文件创建环境
`conda env create -f your_env.yaml`

上面的命令只会导出使用conda安装的，而pip安装的还需要下面的命令
`pip freeze > requirements.txt`

导入pip安装的包
`pip install -r requirements.txt`

还有一种方法直接将**整个环境**内容打包导出，
需要先安装打包工具
`conda install -c conda-forge conda-pack` # 如果安装不了，就去掉-c conda-forge


将环境打包，默认放在C盘用户目录，可以通过在环境名前加路径修改位置
`conda pack -n env_name -o your_out_env.tar.gz`

切换到新电脑上，在Anaconda文件里的envs中创建好新环境目录 your_out_env

解压环境，解压时将your_out_env.tar.gz也放在新环境的文件夹里
cd 对应文件的路径
`tar -xzvf your_out_env.tar.gz`

新电脑激活环境
`conda info -e`  # 查看是否存在了新的环境
`conda activate env_name`




目前已安装的环境 (Local Users, 列举), 及其位置和python 版本。
```
RL                     D:\Anaconda_envs\envs\RL                  Python 3.10.18
drones                 D:\Anaconda_envs\envs\drones              Python 3.10.18
base                   D:\anaconda3                              Python 3.13.5
```
**关键注意事项：（最好）gym==0.21.0只能在pip==24.0版本下安装成功，可以先在pip==24.0环境下成功安装gym,然后升级pip**

```bash
pip install scikit-learn=1.3.2  # sklearn 包的安装
```


### (1) Open-gym or gymnasium as gym

需要安装的数据包：
```bash
pip install mujoco
conda install gymnasium
```

目前主要配置的是游戏环境是：**Open-gym** 游戏环境。
使用`import gym` 或者`import gymnasium as gym`导入Open-AI gym的包。

```
# ----------------------------------------------------------------------------------------------------
# 下面这些环境通常用于测试连续控制算法（如 PPO、SAC、DDPG）。
# ----------------------------------------------------------------------------------------------------
env = gym.make("Ant-v5", render_mode="human")  # 四足蚂蚁机器人，学习行走和转向
env = gym.make("HalfCheetah-v5", render_mode="human")  # 两足猎豹机器人，学习高速奔跑
env = gym.make("Hopper-v4", render_mode="human")  # 单腿跳跃机器人，学习平衡和跳跃
env = gym.make("Humanoid-v5", render_mode="human")  # 3D 人体模型，学习行走和平衡
env = gym.make("HumanoidStandup-v4", render_mode="human")  # 人体模型从躺姿站立
env = gym.make("InvertedDoublePendulum-v4", render_mode="human")  # 双倒立摆，测试控制稳定性
env = gym.make("InvertedPendulum-v4", render_mode="human")  # 经典倒立摆任务（CartPole 的连续控制版）
env = gym.make("Pusher-v5", render_mode="human")  # 机械臂推动物体到目标位置 `Pusher-v4` is only supported on `mujoco<3`
env = gym.make("Reacher-v4", render_mode="human")  # 2-DoF 机械臂到达目标点
env = gym.make("Swimmer-v4", render_mode="human")  # 蛇形游泳机器人，学习在水中游动
env = gym.make("Walker2d-v4", render_mode="human")  # 两足步行机器人，学习行走
# ----------------------------------------------------------------------------------------------------
# 下面这些环境通常用于高级强化学习研究，如分层 RL、多任务学习等。注意：目前的版本不支持以下的游戏环境!!!!!!!!!!!!!!!!!!!
# ----------------------------------------------------------------------------------------------------
# env = gym.make("FetchPickAndPlace-v3", render_mode="human")  # Fetch 机械臂抓取并放置物体（适合机械臂控制）
# env = gym.make("FetchPush-v3", render_mode="human")  # Fetch 机械臂推动物体
# env = gym.make("FetchReach-v3", render_mode="human")  # Fetch 机械臂到达目标位置
# env = gym.make("FetchSlide-v3", render_mode="human")  # Fetch 机械臂滑动物体到目标位置
# env = gym.make("HandManipulateBlock-v1", render_mode="human")  # Shadow Hand 机械手操控方块
# env = gym.make("HandManipulateEgg-v1", render_mode="human")  # Shadow Hand 机械手操控鸡蛋
# env = gym.make("HandManipulatePen-v1", render_mode="human")  # Shadow Hand 机械手操控笔
```
环境 gym 或者 gymnasium 中执行步骤后返回的参数如下所示：
```python
print("gym: env.step(action)返回参数", {
    "observation state(Any)": "执行动作后，环境返回的新状态（观测值），通常是 np.array 或 dict（取决于环境）",
    "reward(float)": "执行动作后获得的即时奖励，用于训练智能体.",
    "terminated(bool)": "True 表示环境因任务正常终止（如游戏胜利/失败）",
    "truncated(bool)": "True 表示环境因外部条件提前终止（如超出最大步数）",
    "info (dict)": "包含额外信息的字典，如调试数据（不用于训练）"
})
```


### (2). [gym-retro (retro_roms)](Env%2Fretro_roms)
gym-retro支持许多任天堂、gym游戏作为RL的学习环境，其中支持的游戏列表的环境如下所示，以下为已经导入Rom.nes文件的游戏列表。
目前主要配置的是游戏环境是：**SuperMarioBros**（超级玛丽 或 超级马里奥）游戏环境。
**重要：retro 仅支持python3.8环境**
进入到conda相关环境后，使用`pip install gym-retro`命令安装retro 实现超级玛丽游戏环境。
导入游戏ROM：`python -m retro.import`.

```
Importing 8Eyes-Nes
Importing MegaMan-Nes
Importing 1942-Nes
Importing Contra-Nes
Importing SCATSpecialCyberneticAttackTeam-Nes   # 类似魂斗罗
Importing Jackal-Nes
Importing ChoujikuuYousaiMacross-Nes
Importing AdventureIslandII-Nes
Importing AdventureIslandII-Nes   # 葫芦岛
Importing AddamsFamilyPugsleysScavengerHunt-Nes
Importing AdventuresOfDinoRiki-Nes
Importing AdventuresOfRockyAndBullwinkleAndFriends-Nes
Importing AdventureIsland3-Nes   # 葫芦岛
Importing Airstriker-Genesis
Importing Amagon-Nes
Importing Arkanoid-Nes
Importing ArkistasRing-Nes
Importing Astyanax-Nes
Importing AttackAnimalGakuen-Nes
Importing AttackOfTheKillerTomatoes-Nes
Importing Ikari-Nes
Importing TeenageMutantNinjaTurtles-Nes
Importing BadDudes-Nes
Importing BadStreetBrawler-Nes
Importing BananaPrince-Nes
Importing Barbie-Nes
Importing Battletoads-Nes
Importing BinaryLand-Nes
Importing SuperMarioBros-Nes  # 超级马里奥兄弟
Importing AdventureIslandII-Nes
Importing AdventureIsland3-Nes
Importing Airstriker-Genesis
Importing StarWars-Nes
Importing SuperMarioBros3-Nes
Importing TeenageMutantNinjaTurtles-Nes
Importing TeenageMutantNinjaTurtlesIITheArcadeGame-Nes
Importing TeenageMutantNinjaTurtlesIIITheManhattanProject-Nes
```
以上为所有导入的retro游戏环境，目前主要应用的是**SuperMarioBros-Nes**游戏（超级马里奥）。[点击返回目录](#目录)

1. 查看支持的游戏列表`python print(retro.data.list_games())`, 注意：未导入ROM.nes文件的游戏环境不可用。
2. 替换为你的游戏名，查看指定游戏可用的状态 `python print(retro.data.list_states('SuperMarioBros-Nes'))`.
3. 使用如下程序可创建环境，并实现可视化。

```python
import retro
import time
# 创建环境
env = retro.make(
    game='SuperMarioBros-Nes',  # ROM名称
    state='',           # 初始关卡
    use_restricted_actions=retro.Actions.ALL,  # 动作空间类型
)
# 查看环境信息
print(f"动作空间: {env.action_space.shape}")
print(f"观察空间: {env.observation_space.shape}")
env.reset()
env.render('human', False)  # 手动开启渲染
for i in range(10000):
    observation, reward, done, info = env.step(env.action_space.sample())
    # print(reward)
    time.sleep(0.01)
    env.render('human', False)
env.close()
```




### (3). [gym-pybullet-drones](Env%2Fgym-pybullet-drones)






## 3. RL的复现和再生产


一个已有 python 库可以实现RL中基础的算法：该库中的算法支持 `gymnasium` 创建的游戏环境。
```python
from stable_baselines3 import TD3, PPO, SAC
import gymnasium as gym
model_TD3 = TD3("MlpPolicy", env=gym.make("Humanoid-v5", render_mode="human"))
model_PPO = PPO("MlpPolicy", env=gym.make("Humanoid-v5", render_mode="human"))
model_SAC = SAC("MlpPolicy", env=gym.make("Humanoid-v5", render_mode="human"))
model_TD3.learn(total_timesteps=1000000)
model_PPO.learn(total_timesteps=1000000)
model_SAC.learn(total_timesteps=1000000)
print(TD3)
```



### Papers Related to the Deep Reinforcement Learning
[01] [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)  
[02] [The Beta Policy for Continuous Control Reinforcement Learning](https://www.ri.cmu.edu/wp-content/uploads/2017/06/thesis-Chou.pdf)  
[03] [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)  
[04] [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)  
[05] [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)  
[06] [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)  
[07] [Continuous Deep Q-Learning with Model-based Acceleration](https://arxiv.org/abs/1603.00748)  
[08] [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)  
[09] [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)  
[10] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[11] [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)  
[12] [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)  
[13] [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)  
[14] [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)  



### DQN 相关的论文

1. Playing Atari with Deep Reinforcement Learning [[arxiv]](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/1.dqn.ipynb)
2. Deep Reinforcement Learning with Double Q-learning [[arxiv]](https://arxiv.org/abs/1509.06461) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/2.double%20dqn.ipynb)
3. Dueling Network Architectures for Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1511.06581) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/3.dueling%20dqn.ipynb)
4. Prioritized Experience Replay [[arxiv]](https://arxiv.org/abs/1511.05952) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb)
5. Noisy Networks for Exploration [[arxiv]](https://arxiv.org/abs/1706.10295) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb)
6. A Distributional Perspective on Reinforcement Learning [[arxiv]](https://arxiv.org/pdf/1707.06887.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/6.categorical%20dqn.ipynb)
7. Rainbow: Combining Improvements in Deep Reinforcement Learning [[arxiv]](https://arxiv.org/abs/1710.02298) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/7.rainbow%20dqn.ipynb)
8. Distributional Reinforcement Learning with Quantile Regression [[arxiv]](https://arxiv.org/pdf/1710.10044.pdf) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/8.quantile%20regression%20dqn.ipynb)
9. Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation  [[arxiv]](https://arxiv.org/abs/1604.06057) [[code]](https://github.com/higgsfield/RL-Adventure/blob/master/9.hierarchical%20dqn.ipynb)
10. Neural Episodic Control [[arxiv]](https://arxiv.org/pdf/1703.01988.pdf) [[code]](#)


**RL评价指标**：每个回合的奖励（越多与好），步数（越少越好）。
以下详细描述了各种RL算法，包括 **训练环境、执行方法、原理** 等。


### (1) [TD3, Twin Delayed Deep Deterministic Policy Gradients](RL%2TD3)


**Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)** [Paper](https://arxiv.org/abs/1802.09477)

[点击返回目录](#目录)

### (2) [OurTD3](RL%2FOurTD3)

待做计划：多进程多步TD3算法。

**目前的TD3算法存在着陷入局部最优的问题，适当增大动作探索的空间，解决局部最优的问题**

在OurTD3文件中，我使用了**DNN、Attention、Encoder、Transformer**重构了TD3中的Actor和Critic架构，可是在Open AI gym 中并没有表现出更好的
性能。运行程序[run_experiments.py](RL%2FOurTD3%2Frun_experiments.py)可以复现TD3在线RL算法。

```bash
cd D:\iResearch\iModel\InverseReinforcementLearning
cd RL/OurTD3
python run_experiments.py
```

### (3). [A3C-Super-mario-bros](RL%2FA3C-Super-mario-bros)

复现 [A3C](https://arxiv.org/abs/1602.01783) 算法在超级玛丽环境的RL，其中超级玛丽环境使用的gym-retro的环境，运行如下代码可以训练A3C在超级玛丽环境`SuperMarioBros-Nes`，**主线程
一个，训练线程6个，测试线程1个**。

```bash
cd D:\iResearch\iModel\InverseReinforcementLearning
cd RL/A3C-Super-mario-bros
python train.py
```

<p align="center">
  <img src="RL/A3C-Super-mario-bros/demo/video_1_1.gif">
  <img src="RL/A3C-Super-mario-bros/demo/video_1_2.gif">
  <img src="RL/A3C-Super-mario-bros/demo/video_1_4.gif"><br/>
  <img src="RL/A3C-Super-mario-bros/demo/video_2_3.gif">
  <img src="RL/A3C-Super-mario-bros/demo/video_3_1.gif">
  <img src="RL/A3C-Super-mario-bros/demo/video_3_4.gif"><br/>
  <img src="RL/A3C-Super-mario-bros/demo/video_4_1.gif">
  <img src="RL/A3C-Super-mario-bros/demo/video_6_1.gif">
  <img src="RL/A3C-Super-mario-bros/demo/video_7_1.gif"><br/>
  <i>训练得到的结果</i>
</p>


### (4) [PG, Policy Gradient](RL%2FPG) 策略梯度算法

PG算法使用的训练环境为 [Open-gym](#1-open-gym) 的游戏环境，


### (5) [PPO-gym](RL%2FPPO-gym) PPO 算法在 [Open-gym](#1-open-gym) 环境上的实现

[PPO论文](https://arxiv.org/abs/1707.06347)

[点击返回目录](#目录)

### (6) [PPO-SuperMarioBros](RL%2FPPO-SuperMarioBros) PPO 算法在 SuperMarioBros 环境上的实现。

### (7) [Conventional Algorithms](RL%2FConventional%20Algorithms) 传统RL算法的实现

* [gridworld （RL的简单通用环境：网格世界的实现）](RL%2FConventional%20Algorithms%2Fgridworld.py)
* [Q-learning](RL%2FConventional%20Algorithms%2FQ-learning.py)
* [Sarsa](RL%2FConventional%20Algorithms%2FSarsa.py)
* [点击返回目录](#目录)



## 4. IRL的复现和再生产


### (1). [MaxEnt](IRL%2FMaxEnt) (最大熵IRL)


运行程序文件[example.py](IRL%2FMaxEnt%2Fsrc%2Fexample.py)可以实现一个MaxEnt 的IRL例子。运行过程如下所示

```bash
cd D:\iResearch\iModel\InverseReinforcementLearning
cd IRL/MaxEnt/src/
python example.py
```


## 5. 多智能体强化学习 （MARL）

### (1) 四种基本的值分解算法

**IQL, VDN, QMIX, QTRAN.**


