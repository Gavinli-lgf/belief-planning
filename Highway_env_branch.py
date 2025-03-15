import pdb
import osqp
import argparse
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import numpy as np
from scipy.io import loadmat
from scipy import interpolate, sparse
import random
import math
from numpy import linalg as LA
from numpy.linalg import norm
from highway_branch_dyn import *

v0=20
f0 = np.array([v0,0,0,0])
lane_width = 3.6
lm = np.arange(0,7)*lane_width  # 用于用于生成车道


def with_probability(P=1):
    return np.random.uniform() <= P

class vehicle():
    def __init__(self, state=[0,0,v0,0],v_length=4,v_width=2.4,dt=0.05,backupidx=0,laneidx=0):
        self.state = np.array(state)
        self.dt = dt
        self.v_length = v_length
        self.v_width = v_width
        self.x_pred = []
        self.y_pred = []
        self.xbackup = None
        self.backupidx = backupidx
        self.laneidx = laneidx
    def step(self,u): # controlled vehicle
        dxdt = np.array([self.state[2]*np.cos(self.state[3]),self.state[2]*np.sin(self.state[3]),u[0],u[1]])
        self.state = self.state + dxdt*self.dt


"""该类是一个高速公路多车辆协同控制仿真环境,用于sim_overtake
"""
class Highway_env():
    def __init__(self,NV,mpc,N_lane=6):
        '''
        Input: NV: number of vehicles(所有车辆数量)
               mpc: mpc controller for the controlled vehicle(ego agent的控制器)
               N_lane: number of lanes
        '''
        self.dt = mpc.predictiveModel.dt    # 仿真步长(与mpc步长一致)
        self.veh_set = []   # 所有车辆对象的集合
        self.NV = NV        # 所有车辆数量
        self.N_lane = N_lane
        self.desired_x = [None]*NV  # 所有车辆期望状态:其中l坐标是veh[i]所在车道中心线的l坐标(如果veh[i]刚进入这个车道,但l坐标还为到达这里，那么这个坐标就是个期望(desire))
        self.mpc = mpc
        self.predictiveModel = mpc.predictiveModel
        self.backupcons = mpc.predictiveModel.backupcons

        self.m = len(self.backupcons)
        self.cons = mpc.predictiveModel.cons
        self.LB = [self.cons.W/2, N_lane*3.6-self.cons.W/2]

        # 初始化所有车辆状态(s,l,v,ψ)(2辆车,生成 x0 为 2*4 的矩阵. ego为第0行，obs为第2行)
        x0 = np.array([[0,1.8,v0,0],[5,5.4,v0,0]])
        # x0 = np.array([[-8,1.8,v0,0],[5,5.4,v0,0]])
        for i in range(0,self.NV):
            self.veh_set.append(vehicle(x0[i],dt=self.dt,backupidx = 0))
            self.desired_x[i] = np.array([0,  x0[i,1],v0,0])


    """
    开始
    ├─ 初始化控制容器
    ├─ 遍历所有车辆
    │   ├─ 生成预测轨迹
    │   ├─ 计算新车道索引
    │   └─ 条件更新车道属性
    ├─ 非控车辆随机变道
    ├─ 安全策略评估
    ├─ MPC求解最优控制
    └─ 状态更新
    结束
    """
    def step(self,t_):
        # initialize the trajectories to be propagated forward under the backup policy
        u_set  = [None]*self.NV # 控制指令集合（所有车辆）
        xx_set = [None]*self.NV # 预测轨迹集合
        u0_set = [None]*self.NV # 备用控制策略集合
        x_set  = [None]*self.NV # 车辆状态集合

        umax = np.array([self.cons.am, self.cons.rm])
        # generate backup trajectories
        self.xbackup = np.empty([0,(self.mpc.N+1)*4])
        for i in range(0,self.NV):
            z = self.veh_set[i].state
            xx_set[i] = self.predictiveModel.zpred_eval(z)  # 调用预测模型生成轨迹
            newlaneidx = round((z[1]-1.8)/3.6)  # 计算 veh[i] 当前所在车道序号（3.6是车道宽度,1.8是车道宽度的一半）

            # 触发车道更新条件（"初始时刻"或"veh[i]的车道序号发生变化,且横向位置与新车道边界偏移<1.4m"）
            if t_==0 or (newlaneidx !=self.veh_set[i].laneidx and abs(z[1]-1.8-3.6*newlaneidx)<1.4):
                # update the desired lane
                self.veh_set[i].laneidx = newlaneidx        # 更新车道序号
                self.desired_x[i][1] = 1.8+newlaneidx*3.6   # 设置新车道中心线l坐标
                # 针对obs(i=1)的特殊处理
                if i==1:
                    # 根据 ego 的 laneidx 动态调整 xRef (通过xRef动态调整参考路径，实现柔性变道)
                    if self.veh_set[0].laneidx<self.veh_set[1].laneidx:
                        xRef = np.array([0, 1.8+3.6*(self.veh_set[1].laneidx-1), v0, 0])
                    elif self.veh_set[0].laneidx>self.veh_set[1].laneidx:
                        xRef = np.array([0, 1.8+3.6*(self.veh_set[1].laneidx+1), v0, 0])
                    else:
                        if self.veh_set[1].laneidx>0:
                            xRef = np.array([0, 1.8+3.6*(self.veh_set[1].laneidx-1), v0, 0])
                        else:
                            xRef = np.array([0, 1.8+3.6*(self.veh_set[1].laneidx+1), v0, 0])

                    # # 更新备用策略（包含三种：保持maintain、刹车brake、变道lc）
                    backupcons = [lambda x:backup_maintain(x,self.cons), lambda x:backup_brake(x,self.cons), lambda x:backup_lc(x,xRef)]
                    self.predictiveModel.update_backup(backupcons)

            # obs(i!=0),每10步(t_%10==0)更新一次 (使用 with_probability() 实现obs的随机变道决策)
            if t_%10==0 and i!=0:
                # update the desired lane for the uncontrolled vehicle(obs有50%概率触发变道)
                if with_probability(0.5):
                    if self.veh_set[i].laneidx==0:  # obs在最左侧车道0,就换到车道1(1.8+3.6=5.4)
                        self.desired_x[i][1] = 5.4
                    elif self.veh_set[i].laneidx==self.N_lane-1:    # obs在最右侧车道(序号N_lane-1),就换到右侧第2车道(N_lane-2)
                        self.desired_x[i][1] = 1.8+(self.N_lane-2)*3.6
                    else:
                        # obs在中间车道(非最左/右侧),就50%概率随机选择向左/右车道
                        if with_probability(0.5):
                            self.desired_x[i][1] = 1.8+(self.veh_set[i].laneidx-1)*3.6  # 向左变道
                        else:
                            self.desired_x[i][1] = 1.8+(self.veh_set[i].laneidx+1)*3.6  # 向右变道

        idx0 = self.veh_set[0].backupidx
        n = self.predictiveModel.n
        x1 = xx_set[0][:, idx0*n:(idx0+1)*n]
        for i in range(0, self.NV):
            if i!=0:    # 对其他 obs 车辆的每个 backupcons 进行安全评估
                hi = np.zeros(self.m)
                for j in range(0, self.m):
                    # 计算安全指数（碰撞检测 + 车道边界约束）
                    hi[j] = min(np.append(veh_col(x1, xx_set[i][:,j*n:(j+1)*n], [self.cons.L+1,self.cons.W+0.2]), lane_bdry_h(x1,self.LB[0],self.LB[1])))
                self.veh_set[i].backupidx = np.argmax(hi)   # 选择最安全的备用策略索引

            # 应用备用策略生成控制指令
            u0_set[i]=self.backupcons[self.veh_set[i].backupidx](self.veh_set[i].state)


        # set x_ref for the overtaking maneuver and call the MPC(根据ego与obs的相对位置关系，设置xRef,并求解mpc.)
        # 根据 ego 与 obs 的纵向相对位置,设置 ego 的期望l坐标 Ydes
        if self.veh_set[0].state[0]<self.veh_set[1].state[0]:   # ego 车辆在 obs 车辆前面
            Ydes = 1.8+self.veh_set[0].laneidx*3.6  # 计算 ego 所在车道中心线的 l 坐标(也是期望l坐标)
        else:   # ego 在 obs 前面时,ego的期望l坐标就是obs当前所处车道的中心线的l坐标
            Ydes = self.veh_set[1].state[1]

        # 根据 ego 是否已经正常对 obs 完成overtake,设置 ego 的期望速度 vdes   
        if abs(self.veh_set[0].state[1]-Ydes)<1 and self.veh_set[0].state[0]>self.veh_set[1].state[0]+3:
            vdes = v0
        else:
            vdes = self.veh_set[1].state[2] + 1*(self.veh_set[1].state[0]+1.5-self.veh_set[0].state[0])

        # Ydes = 1.8+self.veh_set[0].laneidx*3.6
        # vdes = self.veh_set[1].state[2]+5
        xRef = np.array([0,Ydes,vdes,0])
        self.mpc.solve(self.veh_set[0].state, self.veh_set[1].state, xRef)

        u_set[0] = self.mpc.uPred[0]
        xPred, zPred, uPred, branch_w = self.mpc.BT2array()
        self.veh_set[0].step(u_set[0])
        x_set[0] = self.veh_set[0].state
        # if t_==50:
        #     plot_snapshot(self.veh_set[0].state,self.veh_set[1].state,self,idx=None,varycolor=True,zpatch=False,arrow = False,legend = True)
        # if t_==25 or t_==35 or t_==50:
        #     plot_snapshot(self.veh_set[0].state,self.veh_set[1].state,self,idx=None,varycolor=True,zpatch=False,arrow = False,legend = False)

        for i in range(1,self.NV):
            u_set[i] = u0_set[i]
            self.veh_set[i].step(u_set[i])
            x_set[i] = self.veh_set[i].state

        return u_set, x_set, xx_set, xPred,zPred, branch_w

    def replace_veh(self,idx,dir = 2):
        if idx==0:
            return
        if dir ==0:
            UB = self.veh_set[0].state[0]+13
            LB = self.veh_set[0].state[0]+8
        elif dir==1:
            UB = self.veh_set[0].state[0]-5
            LB = self.veh_set[0].state[0]-13
        else:
            UB = self.veh_set[0].state[0]+15
            LB = self.veh_set[0].state[0]-15

        if self.veh_set[0].laneidx==0:
            laneidx = 1
        elif self.veh_set[0].laneidx==self.N_lane-1:
            laneidx = self.N_lane-2
        else:
            if with_probability(0.5):
                laneidx = self.veh_set[0].laneidx-1
            else:
                laneidx = self.veh_set[0].laneidx+1
        success = False
        count = 0
        while not success:
            count+=1
            Y = (laneidx+0.5)*lane_width+np.random.normal(0,0.1)
            X = random.random()*(UB-LB)+LB
            collision = False
            for i in range(0,self.NV):
                if i!=idx:
                    if abs(Y-self.veh_set[i].state[1])<=2.2 and abs(X-self.veh_set[i].state[0])<=5:
                        collision=True
                        break
            if not collision:
                success = True
            if count>20:
                return False
        self.veh_set[idx] = vehicle([X,Y,self.veh_set[0].state[2],0],dt=self.dt,backupidx = 0,laneidx = laneidx)
        return True

def merge_geometry(N_lane,merge_lane,merge_s,merge_R, merge_side=0):
    '''
    generate the merging geometry
    input: N_lane: number of lanes on the main highway
           merge_lane: number of lanes on the ramp
           merge_s: X coordinate of the merging position
           merge_R: radius of the arc for the ramp
           merge_side: left or right
    '''
    merge_theta = np.arccos(1-lane_width*merge_lane/merge_R)

    merge_end = merge_s+merge_R*np.sin(merge_theta)
    if merge_side==0:
        merge_arc_center = np.array([merge_s+merge_R*np.sin(merge_theta),(N_lane-merge_lane)*lane_width+merge_R])
        merge_lane_start = np.array([merge_s-merge_s*np.cos(merge_theta),N_lane*lane_width+np.sin(merge_theta)*merge_s])
    else:
        merge_arc_center = np.array([merge_s+merge_R*np.sin(merge_theta),merge_lane*lane_width-merge_R])
        merge_lane_start = np.array([merge_s-merge_s*np.cos(merge_theta),-np.sin(merge_theta)*merge_s-lane_width*merge_lane])


    merge_lane_ref_s1 = np.linspace(0, merge_s, num=int(merge_s/0.5),endpoint = False) # straight portion
    merge_lane_ref_s2 = merge_s + np.linspace(0,merge_R*merge_theta,num=int(merge_R*merge_theta/0.5))  #arc portion
    if merge_side==0:
        merge_lane_ref_X1 = merge_lane_start[0]+merge_lane_ref_s1*np.cos(merge_theta)
        merge_lane_ref_Y1 = merge_lane_start[1]-merge_lane_ref_s1*np.sin(merge_theta)
        merge_lane_ref_psi1 = -np.ones(merge_lane_ref_s1.shape)*merge_theta
        merge_lane_ref_psi2 = (merge_lane_ref_s2-merge_lane_ref_s2[-1])/merge_R
        merge_lane_ref_X2 = merge_arc_center[0] + np.sin(merge_lane_ref_psi2)*merge_R
        merge_lane_ref_Y2 = merge_arc_center[1] - np.cos(merge_lane_ref_psi2)*merge_R
    else:
        merge_lane_ref_X1 = merge_lane_start[0]+merge_lane_ref_s1*np.cos(merge_theta)
        merge_lane_ref_Y1 = merge_lane_start[1]+merge_lane_ref_s1*np.sin(merge_theta)
        merge_lane_ref_psi1 = np.ones(merge_lane_ref_s1.shape)*merge_theta
        merge_lane_ref_psi2 = (merge_lane_ref_s2[-1]-merge_lane_ref_s2)/merge_R
        merge_lane_ref_X2 = merge_arc_center[0] - np.sin(merge_lane_ref_psi2)*merge_R
        merge_lane_ref_Y2 = merge_arc_center[1] + np.cos(merge_lane_ref_psi2)*merge_R-merge_lane*lane_width


    return merge_lane_ref_X1,merge_lane_ref_X2,merge_lane_ref_Y1,merge_lane_ref_Y2,merge_lane_ref_psi1,merge_lane_ref_psi2



"""该类是一个高速公路多车辆协同控制仿真环境,用于sim_merge
"""
class Highway_env_merge():
    '''
    Similar object, for merging simulation
    '''
    def __init__(self,NV,N_lane, mpc, pred_model, merge_lane=2,merge_s = 50,merge_R=300, merge_side = 0, dt=0.05):
        self.dt = dt
        self.veh_set = []
        self.NV = NV
        self.laneID = [1]+[0]*(NV-1)
        self.N_lane = N_lane
        self.merge_lane = merge_lane
        self.desired_x = [None]*NV
        self.merge_s = merge_s
        self.merge_R = merge_R
        self.merge_side = merge_side
        self.pred_model = pred_model
        self.mpc = mpc
        self.m = [None]*len(pred_model)
        self.backupcons = [None]*len(pred_model)
        for i in range(0,len(pred_model)):
            self.backupcons[i] = self.pred_model[i].backupcons
            self.m[i] = len(self.backupcons[i])
        self.cons = mpc.predictiveModel.cons
        self.LB = [self.cons.W/2,N_lane*3.6-self.cons.W/2]

        merge_lane_ref_X1,merge_lane_ref_X2,merge_lane_ref_Y1,merge_lane_ref_Y2,merge_lane_ref_psi1,merge_lane_ref_psi2 = merge_geometry(N_lane,merge_lane,merge_s,merge_R, merge_side)
        self.merge_theta = np.arccos(1-lane_width*merge_lane/merge_R)
        self.merge_end = merge_s+merge_R*np.sin(self.merge_theta)
        self.merge_lane_ref_Y = np.append(merge_lane_ref_Y1,merge_lane_ref_Y2)
        self.merge_lane_ref_X = np.append(merge_lane_ref_X1,merge_lane_ref_X2)
        self.merge_lane_ref_psi = np.append(merge_lane_ref_psi1,merge_lane_ref_psi2)
        self.merge_lane_ref_Y1 = merge_lane_ref_Y1
        self.merge_lane_ref_Y2 = merge_lane_ref_Y2
        self.merge_lane_ref_X1 = merge_lane_ref_X1
        self.merge_lane_ref_X2 = merge_lane_ref_X2
        self.merge_lane_ref_psi1 = merge_lane_ref_psi1
        self.merge_lane_ref_psi2 = merge_lane_ref_psi2
        self.refY = interpolant('refY','linear',[self.merge_lane_ref_X],self.merge_lane_ref_Y)
        self.refpsi = interpolant('refY','linear',[self.merge_lane_ref_X],self.merge_lane_ref_psi)
        UB = 30
        LB = 0
        x0 = np.array([[24,13,v0,-0.2],[15,5.4,v0,0]])
        for i in range(0,self.NV):
            self.veh_set.append(vehicle(x0[i],dt=self.dt,backupidx = 0))
            self.desired_x[i] = np.array([0,  x0[i,1],v0,0])

    def step(self,t_):
        u_set  = [None]*self.NV
        xx_set = [None]*self.NV
        u0_set = [None]*self.NV
        x_set  = [None]*self.NV

        umax = np.array([self.cons.am,self.cons.rm])
        # generate backup trajectories
        self.xbackup = np.empty([0,(self.mpc.N+1)*4])
        for i in range(0,self.NV):
            z = self.veh_set[i].state
            if self.veh_set[i].state[0]>self.merge_s+8:
                self.laneID[i] = 0
            xx_set[i] = self.pred_model[self.laneID[i]].zpred_eval(z)


        idx0 = self.veh_set[0].backupidx
        n = self.pred_model[self.laneID[0]].n
        x1 = xx_set[0][:,idx0*n:(idx0+1)*n]
        for i in range(0,self.NV):
            if i!=0:
                hi = np.zeros(self.m[self.laneID[i]])
                if self.laneID[i]==0:
                    for j in range(0,self.m[0]):
                        hi[j] = min(np.append(veh_col(x1,xx_set[i][:,j*n:(j+1)*n],[self.cons.L+1,self.cons.W+0.2]),lane_bdry_h(xx_set[i][:,j*n:(j+1)*n],self.LB[0],self.LB[1])))
                elif self.laneID[i]==1:
                    for j in range(0,self.m[1]):
                        hi[j] = veh_col(x1,xx_set[i][:,j*n:(j+1)*n],[self.cons.L+1,self.cons.W+0.2])
                self.veh_set[i].backupidx = np.argmax(hi)
            self.veh_set[i].backupidx = 0
            u0_set[i]=self.backupcons[self.laneID[i]][self.veh_set[i].backupidx](self.veh_set[i].state)


        x = self.veh_set[0].state
        if self.laneID[0]==0:
            S = np.eye(4)
            xRef = np.array([0,(self.N_lane-0.5)*3.6,v0,0])
            bx = self.mpc.param.bx

        else:
            y0 = float(self.refY(x[0]))
            psi0 = float(self.refpsi(x[0]))
            S = np.array([[1.,0,0,0],[-np.tan(psi0),1.,0,0],[0,0,1,0],[0,0,0,1]])
            xRef = np.array([0,-np.tan(psi0)*x[0]+y0+1.8,v0,psi0])
            bx = np.array([-np.tan(psi0)*x[0]+y0+3.6*self.merge_lane-self.cons.W/2,np.tan(psi0)*x[0]-y0-self.cons.W/2,psi0+self.mpc.psimax,-psi0+self.mpc.psimax])


        self.mpc.solve(self.veh_set[0].state,self.veh_set[1].state,xRef,S,Fx=None,bx=bx)



        u_set[0] = self.mpc.uPred[0]
        xPred,zPred,uPred,branch_w = self.mpc.BT2array()
        self.veh_set[0].step(u_set[0])
        x_set[0] = self.veh_set[0].state


        for i in range(1,self.NV):
            u_set[i] = u0_set[i]
            self.veh_set[i].step(u_set[i])
            x_set[i] = self.veh_set[i].state


        return u_set,x_set,xx_set,xPred,zPred,branch_w

"""
输出:(注: 运行结果图中y显示的是负数,但是代码中所有变量的y都是按正数记录的)
    state_rec: 车辆状态记录 (共3个维度[车辆序号, 时间步, 状态向量列表])
    input_rec: 控制输入记录 (共3个维度[车辆编号, 时间步, 控制向量列表])
    backup_rec: 各车辆的备份轨迹 (共3个维度[车辆序号, 时间步, 状态向量列表])
    backup_choice_rec: 各车辆的备份策略索引(共3个维度[车辆序号, 时间步, 策略索引列表]
    xPred_rec: ego 的预测轨迹记录(共3个维度[时间步, 第几条预测线序号, 每条预测线上的状态序号, 状态向量列表])
    zPred_rec: obs 的预测轨迹记录(共3个维度,与xPred_rec相同)
"""
def Highway_sim(env,T):
    # simulate the scenario
    collision = False
    dt = env.dt                           # 从 env 对象获取时间步长
    t=0
    Ts_update = 4
    N_update = int(round(Ts_update/dt))
    N = int(round(T/dt))                  # 总仿真步数
    state_rec = np.zeros([env.NV, N, 4])  # 车辆状态记录 [车辆编号, 时间步, 状态向量]
    b_rec = [None]*N
    backup_rec = [None]*env.NV          # 各车辆的备份轨迹
    backup_choice_rec = [None]*env.NV   # 备份策略选择索引
    xPred_rec = [None]*N
    zPred_rec = [None]*N
    branch_w_rec = [None]*N
    for i in range(0,env.NV):
        backup_rec[i]=[None]*N
        backup_choice_rec[i] = [None]*N
    input_rec = np.zeros([env.NV, N, 2])     # 控制输入记录 [车辆编号, 时间步, 控制向量]
    f0 = np.array([v0,0,0,0])
    for i in range(0,len(env.veh_set)):
        state_rec[i][t]=env.veh_set[i].state

    # 主仿真循环(步长dt,总时间T,总仿真步数N)
    xx_set = []
    dis = 100
    while t<N:  # t 是仿真步数
        # 碰撞检测模块
        if not collision:
            for i in range(0, env.NV):
                for j in range(0, env.NV):
                    if i!=j:
                        # 计算车辆间距（考虑车体尺寸）
                        dis = max(abs(env.veh_set[i].state[0]-env.veh_set[j].state[0])-0.5*(env.veh_set[i].v_length+env.veh_set[j].v_length),\
                        abs(env.veh_set[i].state[1]-env.veh_set[j].state[1])-0.5*(env.veh_set[i].v_width+env.veh_set[j].v_width))
                if dis<0:   # 发生碰撞
                    collision = True

        print("t=",t*env.dt)

        # 调用 env 对象,计算第 t 步仿真结果
        u_set,x_set,xx_set,xPred,zPred,branch_w=env.step(t)
        xPred_rec[t]=xPred
        zPred_rec[t]=zPred
        branch_w_rec[t] = branch_w
        # 记录关键数据
        for i in range(0, env.NV):
            input_rec[i][t]=u_set[i]    # 记录 t 步的控制指令
            state_rec[i][t]=x_set[i]    # 记录 t 步的车辆状态
            backup_rec[i][t]=xx_set[i]  # 记录 t 步的备份轨迹
            backup_choice_rec[i][t] = env.veh_set[i].backupidx  # 记录第 t 步选择的策略索引
        t=t+1
    return state_rec,input_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,branch_w_rec,collision

def plot_snapshot(x,z,env,idx=None,varycolor=True,zpatch=True,arrow = True,legend = True):
    '''
    Ploting a snapshot of the simulation, for debugging
    '''
    plot_merge = isinstance(env,Highway_env_merge)
    if plot_merge:
        fig = plt.figure(figsize=(15,6))
    else:
        fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)
    ego_idx = 0
    ego_veh = env.veh_set[ego_idx]
    veh_patch = [None]*env.NV
    for i in range(0,env.NV):
        if i==ego_idx:
            veh_patch[i]=plt.Rectangle((ego_veh.state[0]-ego_veh.v_length/2,ego_veh.state[1]-ego_veh.v_width/2), ego_veh.v_length,ego_veh.v_width, fc='r', zorder=0)
        else:
            veh_patch[i]=plt.Rectangle((ego_veh.state[0]-ego_veh.v_length/2,ego_veh.state[1]-ego_veh.v_width/2), ego_veh.v_length,ego_veh.v_width, fc='b', zorder=0)


    ego_y = ego_veh.state[1]
    ego_x = ego_veh.state[0]
    if plot_merge:
        xmin = ego_x-10
        xmax = ego_x+40
        ymin = -5
        ymax = 15
    else:
        xmin = ego_x-10
        xmax = ego_x+40
        ymin = ego_y-5
        ymax = ego_y+10
    try:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-ymax,-ymin )
    except:
        pdb.set_trace()
    ts = ax.transData
    for i in range(0,env.NV):
        coords = ts.transform([env.veh_set[i].state[0],-env.veh_set[i].state[1]])
        tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -env.veh_set[i].state[3])
        te= ts + tr
        veh_patch[i].set_xy([env.veh_set[i].state[0]-env.veh_set[i].v_length/2,-env.veh_set[i].state[1]-env.veh_set[i].v_width/2])
        veh_patch[i].set_transform(te)
        ax.add_patch(veh_patch[i])

    colorset = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','y','m','c','g']
    xPred,zPred,uPred,branch_w = env.mpc.BT2array()
    if idx is None:
        idx = range(0,len(zPred))

    for j in idx:
        for k in range(0,xPred[j].shape[0]):
            if k%2==1:
                if varycolor:
                    x_patch = plt.Rectangle((xPred[j][k,0]-env.veh_set[ego_idx].v_length/2,-xPred[j][k,1]-env.veh_set[ego_idx].v_width/2), env.veh_set[ego_idx].v_length,env.veh_set[ego_idx].v_width,ec=colorset[j], fc=colorset[j],alpha=0.2, zorder=0)
                else:
                    x_patch = plt.Rectangle((xPred[j][k,0]-env.veh_set[ego_idx].v_length/2,-xPred[j][k,1]-env.veh_set[ego_idx].v_width/2), env.veh_set[ego_idx].v_length,env.veh_set[ego_idx].v_width,ec='y', fc='y',alpha=0.2, zorder=0)
                coords = ts.transform([xPred[j][k,0],-xPred[j][k,1]])
                if arrow:
                    arr = ax.arrow(xPred[j][k,0],-xPred[j][k,1],uPred[j][k,0]*np.cos(xPred[j][k,3]),-uPred[j][k,0]*np.sin(xPred[j][k,3]),head_width=0.5,length_includes_head=True)
                else:
                    arr = None
                tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -xPred[j][k,3])
                x_patch.set_transform(ts+tr)
                ax.add_patch(x_patch)

    for j in idx:
        z_patch = plt.plot(zPred[j][:,0],-zPred[j][:,1],'m--',linewidth = 2)[0]
        if zpatch:
            for k in range(0,zPred[j].shape[0]):
                if k%2==1:
                    if varycolor:
                        z_patch = plt.Rectangle((zPred[j][k,0]-env.veh_set[1].v_length/2,-zPred[j][k,1]-env.veh_set[1].v_width/2), env.veh_set[1].v_length,env.veh_set[1].v_width,ec=colorset[-1-j], fc=colorset[-1-j],alpha=0.2, zorder=0)
                    else:
                        z_patch = plt.Rectangle((zPred[j][k,0]-env.veh_set[1].v_length/2,-zPred[j][k,1]-env.veh_set[1].v_width/2), env.veh_set[1].v_length,env.veh_set[1].v_width,ec='c', fc='c',alpha=0.2, zorder=0)
                    coords = ts.transform([zPred[j][k,0],-zPred[j][k,1]])
                    tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -zPred[j][k,3])
                    z_patch.set_transform(ts+tr)
                    ax.add_patch(z_patch)
    if legend:
        ax.legend([x_patch,z_patch,arr],['Planned trajectory for ego vehicle','Predicted trajectory for the uncontrolled vehicle','Ego vehicle acceleration'],fontsize=15)
    if plot_merge:
        if env.merge_side==0:
            plt.plot([-10, 1000],[-lm[0], -lm[0]], 'g', linewidth=2)
            for j in range(1, env.N_lane):
                plt.plot([-10, 1000],[-lm[j], -lm[j]],  'g--', linewidth=1)

            plt.plot([-10, env.merge_s],[-lm[env.N_lane], -lm[env.N_lane]],  'g', linewidth=2)
            plt.plot([env.merge_end, 1000],[-lm[env.N_lane], -lm[env.N_lane]],  'g', linewidth=2)

            plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1, 'g', linewidth=2)
            plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2, 'g--', linewidth=1)
            for j in range(1,env.merge_lane):
                plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1-j*lane_width, 'g--', linewidth=1)
                plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2-j*lane_width, 'g--', linewidth=1)
            plt.plot(env.merge_lane_ref_X, -env.merge_lane_ref_Y-env.merge_lane*lane_width, 'g', linewidth=2)
        else:
            plt.plot([-10, 1000],[-lm[env.N_lane], -lm[env.N_lane]],  'g', linewidth=2)
            for j in range(1, env.N_lane):
                plt.plot([-10, 1000],[-lm[j], -lm[j]], 'g--', linewidth=1)

            plt.plot([-10, env.merge_s],[-lm[0], -lm[0]],  'g', linewidth=2)
            plt.plot([env.merge_end, 1000],[-lm[0], -lm[0]],  'g', linewidth=2)

            plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1, 'g', linewidth=2)
            plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2, 'g', linewidth=2)
            for j in range(1,env.merge_lane):
                plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1-j*lane_width, 'g--', linewidth=1)
                plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2-j*lane_width, 'g--', linewidth=1)
            plt.plot(env.merge_lane_ref_X, -env.merge_lane_ref_Y-env.merge_lane*lane_width, 'g', linewidth=2)
    else:
        plt.plot([xmin-50, xmax+50],[-lm[0], -lm[0]], 'g', linewidth=2)
        for j in range(1, env.N_lane):
            plt.plot([xmin-50, xmax+50],[-lm[j], -lm[j]], 'g--', linewidth=1)
        plt.plot([xmin-50, xmax+50],[-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)
    plt.show()


def animate_scenario(env,state_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,lm,output=None):
    '''
    Animate the simulation
    '''
    if output:
        matplotlib.use("Agg")
    ego_idx = 0  # ego agent 编号为0
    # 根据环境类型设置画布尺寸
    plot_merge = isinstance(env,Highway_env_merge)
    if plot_merge:
        fig = plt.figure(figsize=(10,8))    # 合并场景需要更大画布
    else:
        fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111)
    plt.grid()

    # 创建车辆矩形显示 veh_patch
    nframe = len(state_rec[0])
    ego_veh = env.veh_set[ego_idx]
    veh_patch = [None]*env.NV
    for i in range(0, env.NV):
        if i==ego_idx:  # ego agent 用红色表示'r'
            veh_patch[i]=plt.Rectangle((ego_veh.state[0]-ego_veh.v_length/2, ego_veh.state[1]-ego_veh.v_width/2), ego_veh.v_length, ego_veh.v_width, fc='r', zorder=0)
        else:           # 其它 obs 用蓝色表示'b'
            veh_patch[i]=plt.Rectangle((ego_veh.state[0]-ego_veh.v_length/2, ego_veh.state[1]-ego_veh.v_width/2), ego_veh.v_length, ego_veh.v_width, fc='b', zorder=0)

    for patch in veh_patch:
        ax.add_patch(patch)


    """
    animate 实现了每一帧的具体显示效果(可通过实际显示结果反推变量的含义)
    输入:(注: 运行结果图中y显示的是负数,但是代码中所有变量的y都是按正数记录的)
        t: 第几帧(第几步)
        veh_patch: 
        state_rec: 车辆状态记录(共3个维度[车辆序号, 时间步, 状态向量])
        backup_rec: 未使用
        backup_choice_rec: 未使用
        xPred_rec: ego 的预测轨迹记录(共4个维度[时间步, 第几条预测线序号, 每条预测线上的状态序号, 状态向量(x,y,v,ψ)])
        zPred_rec: obs 的预测轨迹记录(共4个维度,与xPred_rec相同)
    """
    def animate(t,veh_patch,state_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,env,ego_idx=0):
        # 根据场景类型动态调整视野范围
        plot_merge = isinstance(env,Highway_env_merge)
        N_veh = len(state_rec)  # 所有车辆数
        ego_y = state_rec[ego_idx][t][1]
        ego_x = state_rec[ego_idx][t][0]
        ax.clear()
        if plot_merge:      # 合并场景的固定视野
            xmin = ego_x-5
            xmax = ego_x+45
            ymin = -5
            ymax = 35
        else:               # 普通场景的跟随视野
            xmin = ego_x-10
            xmax = ego_x+40
            ymin = ego_y-10
            ymax = ego_y+10

        try:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-ymax,-ymin )
        except:
            pdb.set_trace()

        # 绘制 ego 与 obs
        ts = ax.transData
        for i in range(0,N_veh):
            coords = ts.transform([state_rec[i][t][0], -state_rec[i][t][1]])
            tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -state_rec[i][t][3])
            te= ts + tr
            veh_patch[i].set_xy([state_rec[i][t][0]-env.veh_set[i].v_length/2, -state_rec[i][t][1]-env.veh_set[i].v_width/2])
            veh_patch[i].set_transform(te)
            ax.add_patch(veh_patch[i])
            idx = backup_choice_rec[i][t]

        # 绘制 ego 的预测轨迹(colorset 表示每条预测线的颜色集合)
        colorset = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','y','m','c','g']
        for j in range(0, len(xPred_rec[t])):   # 遍历 ego 在 t 时刻所有的预测线
            plt.plot(xPred_rec[t][j][:,0], -xPred_rec[t][j][:,1], 'b--',linewidth = 1)
            for k in range(0,xPred_rec[t][j].shape[0]): # 遍历第j条预测线的所有状态点
                if k%2==1:
                    # 用半透明色块 newpatch 表示预测位置
                    newpatch = plt.Rectangle((xPred_rec[t][j][k,0]-env.veh_set[ego_idx].v_length/2,-xPred_rec[t][j][k,1]-env.veh_set[ego_idx].v_width/2), env.veh_set[ego_idx].v_length,env.veh_set[ego_idx].v_width, fc=colorset[j],alpha=0.3, zorder=0)
                    coords = ts.transform([xPred_rec[t][j][k,0],-xPred_rec[t][j][k,1]])
                    tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -xPred_rec[t][j][k,3])
                    newpatch.set_transform(ts+tr)
                    ax.add_patch(newpatch)

        # 绘制 obs 的预测轨迹
        for j in range(0,len(zPred_rec[t])):
            plt.plot(zPred_rec[t][j][:,0],-zPred_rec[t][j][:,1],'r--',linewidth = 1)

        # 绘制车道线
        if plot_merge:
            if env.merge_side==0:
                plt.plot([-10, 1000],[-lm[0], -lm[0]], 'g', linewidth=2)
                for j in range(1, env.N_lane):
                    plt.plot([-10, 1000],[-lm[j], -lm[j]],  'g--', linewidth=1)

                plt.plot([-10, env.merge_s],[-lm[env.N_lane], -lm[env.N_lane]],  'g', linewidth=2)
                plt.plot([env.merge_end, 1000],[-lm[env.N_lane], -lm[env.N_lane]],  'g', linewidth=2)

                plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1, 'g', linewidth=2)
                plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2, 'g--', linewidth=1)
                for j in range(1,env.merge_lane):
                    plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1-j*lane_width, 'g--', linewidth=1)
                    plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2-j*lane_width, 'g--', linewidth=1)
                plt.plot(env.merge_lane_ref_X, -env.merge_lane_ref_Y-env.merge_lane*lane_width, 'g', linewidth=2)
            else:
                plt.plot([-10, 1000],[-lm[env.N_lane], -lm[env.N_lane]],  'g', linewidth=2)
                for j in range(1, env.N_lane):
                    plt.plot([-10, 1000],[-lm[j], -lm[j]], 'g--', linewidth=1)

                plt.plot([-10, env.merge_s],[-lm[0], -lm[0]],  'g', linewidth=2)
                plt.plot([env.merge_end, 1000],[-lm[0], -lm[0]],  'g', linewidth=2)

                plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1, 'g', linewidth=2)
                plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2, 'g', linewidth=2)
                for j in range(1,env.merge_lane):
                    plt.plot(env.merge_lane_ref_X1, -env.merge_lane_ref_Y1-j*lane_width, 'g--', linewidth=1)
                    plt.plot(env.merge_lane_ref_X2, -env.merge_lane_ref_Y2-j*lane_width, 'g--', linewidth=1)
                plt.plot(env.merge_lane_ref_X, -env.merge_lane_ref_Y-env.merge_lane*lane_width, 'g', linewidth=2)
        else:
            plt.plot([xmin-50, xmax+50],[-lm[0], -lm[0]], 'g', linewidth=2)
            for j in range(1, env.N_lane):
                plt.plot([xmin-50, xmax+50],[-lm[j], -lm[j]], 'g--', linewidth=1)
            plt.plot([xmin-50, xmax+50],[-lm[env.N_lane], -lm[env.N_lane]], 'g', linewidth=2)

        return veh_patch

    anim = animation.FuncAnimation(fig, animate, fargs=(veh_patch,state_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,env,ego_idx,),
                                   frames=nframe,
                                   interval=50,
                                   blit=False, repeat=False)


    if output:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(1/env.dt), metadata=dict(artist='Me'), bitrate=1800)
        anim_name = output
        anim.save(anim_name,writer=writer)
    else:
        plt.show()



def sim_overtake(mpc,N_lane):

    env = Highway_env(NV=2,mpc = mpc,N_lane=N_lane)
    state_rec,input_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,branch_w_rec,collision = Highway_sim(env,10)

    animate_scenario(env,state_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,lm)
    br = np.reshape(np.array(branch_w_rec),[-1,12])

def sim_merge(mpc,pred_model,N_lane,merge_lane,merge_s ,merge_R, merge_side):


    NV = 2
    env = Highway_env_merge(NV,N_lane, mpc, pred_model, merge_lane,merge_s,merge_R, merge_side,pred_model[0].dt)
    state_rec,input_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,branch_w_rec,collision = Highway_sim(env,6)
    animate_scenario(env,state_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,lm)
