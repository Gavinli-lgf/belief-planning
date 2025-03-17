from casadi import *
import pdb
import itertools
import numpy as np
from scipy import sparse
from itertools import product
'''
Use CasADi to calculate the dynamics, propagate trajectories under the backup policies, and calculate the branching probabilities
'''


"""
功能:输入状态x(x,y,v,ψ)与控制输入u(a,r),返回状态导数x_dot(v_x, v_y, a, r).
预备知识:SX与MX都是casadi中矩阵表达式类型的符号变量类型.
    SX用于问题规模较小且需要高效计算;MX用于问题规模较大或表达式复杂.
"""
def dubin(x,u):
    # 如果 x 是 numpy.ndarray 类型，则使用 NumPy 数组计算状态导数。
    # 如果 x 是 casadi的 SX 或 MX 类型，则使用 CasADi 的符号计算状态导数。
    if isinstance(x,numpy.ndarray):
        xdot = np.array([x[2]*cos(x[3]),x[2]*sin(x[3]),u[0],u[1]])
    elif isinstance(x,casadi.SX):
        xdot = SX(4,1)
        xdot[0]=x[2]*cos(x[3])
        xdot[1]=x[2]*sin(x[3])
        xdot[2]=u[0]
        xdot[3]=u[1]
    elif isinstance(x,casadi.MX):
        xdot = MX(4,1)
        xdot[0]=x[2]*cos(x[3])
        xdot[1]=x[2]*sin(x[3])
        xdot[2]=u[0]
        xdot[3]=u[1]
    return xdot


def softsat(x,s):
    return (np.exp(s*x)-1)/(np.exp(s*x)+1)*0.5+0.5


"""
功能:通过比例控制计算r,减速度a取0,实现车辆的维持状态(保持当前速度和方向不变),并输出对应的控制量u
输入:
    x: 当前状态(x,y,v,ψ),类型为numpy.ndarray或casadi.SX或casadi.MX.
    cons: 模型参数整合的对象.
输出:
    u: 计算得到的控制输入(a,r),类型为numpy.ndarray或casadi.SX或casadi.MX.
"""
def backup_maintain(x,cons,psiref=None):
    if psiref is None:
        if isinstance(x,casadi.MX):
            u = MX(2,1)
            u[0] = 0
            u[1] = -cons.Kpsi*x[3]
            return u
        elif isinstance(x,casadi.SX):
            u = SX(2,1)
            u[0] = 0
            u[1] = -cons.Kpsi*x[3]
            return u
        else:
            return np.array([0.,-cons.Kpsi*x[3]])
    else:
        if isinstance(x,casadi.MX):
            u = MX(2,1)
            u[1] = psiref(x[0])-cons.Kpsi*x[3]
            return u
        elif isinstance(x,casadi.SX):
            u = SX(2,1)
            u[1] = psiref(x[0])-cons.Kpsi*x[3]
            return u
        else:
            return np.array([0.,psiref(x[0])-cons.Kpsi*x[3]])

def backup_maintain_trackV(x,cons,v0,psiref=None):
    if psiref is None:
        if isinstance(x,casadi.MX):
            u = MX(2,1)
            u[0] = 0.5*(v0-x[2])
            u[1] = -cons.Kpsi*x[3]
            return u
        else:
            return np.array([0.5*(v0-x[2]),-cons.Kpsi*x[3]])
    else:
        if isinstance(x,casadi.MX):
            u = MX(2,1)
            u[0] = 0.5*(v0-x[2])
            u[1] = psiref(x[0])-cons.Kpsi*x[3]
            return u
        else:
            return np.array([0.5*(v0-x[2]),psiref(x[0])-cons.Kpsi*x[3]])


"""
功能:通过softmax计算a,比例控制计算r,输出车辆能够加速/刹车的控制量u
输入:
    x: 当前状态(x,y,v,ψ),类型为numpy.ndarray或casadi.SX或casadi.MX.
    cons: 模型参数整合的对象.
输出:
    u: 计算得到的控制输入(a,r),类型为numpy.ndarray或casadi.SX或casadi.MX.
"""
def backup_brake(x,cons,psiref=None):
    if psiref is None:
        if isinstance(x,casadi.MX):
            u = MX(2,1)
            u[0] = softmax(vertcat(-7,-x[2]),5) #根据当前车辆速度 x[2]，计算一个合理的加速度控制量 u[0]，使车辆能够减速或刹车。
            u[1] = -cons.Kpsi*x[3]
            return u
        elif isinstance(x,casadi.SX):
            u = SX(2,1)
            u[0] = softmax(vertcat(-7,-x[2]),5)
            u[1] = -cons.Kpsi*x[3]
            return u
        else:
            return np.array([softmax(vertcat(-5,-x[2]),3),-cons.Kpsi*x[3]])
    else:
        if isinstance(x,casadi.MX) or isinstance(x,casadi.SX):
            u = 0.*x[0:2]
            u[0] = softmax(vertcat(-5,-x[2]),3)
            u[1] = psiref(x[0])-cons.Kpsi*x[3]
            return u

        else:
            return np.array([softmax(vertcat(-5,-x[2]),3),psiref(x[0])-cons.Kpsi*x[3]])

"""
计算车辆在换道从x0到x时的控制输入 u. 
其中加速度 u[0] 通过比例控制; 转向角速度 u[1] 通过比例控制调整横向位置和航向角.
"""
def backup_lc(x,x0):
    if isinstance(x,casadi.MX):
        u = MX(2,1)
        u[0] = -0.8558*(x[2]-x0[2])
        u[1] = -0.3162*(x[1]-x0[1])-3.9889*(x[3]-x0[3])
        return u
    elif isinstance(x,casadi.SX):
        u = SX(2,1)
        u[0] = -0.8558*(x[2]-x0[2])
        u[1] = -0.3162*(x[1]-x0[1])-3.9889*(x[3]-x0[3])
        return u
    else:
        return np.array([-0.8558*(x[2]-x0[2]),-0.3162*(x[1]-x0[1])-3.9889*(x[3]-x0[3])])

# 通过指数加权的方式计算输入向量的软最小值，结果介于 x 的最小值和最大值之间。(适用于分支预测控制中的安全性评估和平滑处理。)
def softmin(x,gamma=1):
    if isinstance(x,casadi.SX) or isinstance(x,casadi.MX):
        return sum1(exp(-gamma*x)*x)/sum1(exp(-gamma*x))
    else:
        return np.sum(np.exp(-gamma*x)*x)/np.sum(np.exp(-gamma*x))

# 确保减速度的变化是平滑的，避免突然的剧烈变化。
def softmax(x,gamma=1):
    if isinstance(x,casadi.SX) or isinstance(x,casadi.MX):
        return sum1(exp(gamma*x)*x)/sum1(exp(gamma*x))
    else:
        return np.sum(np.exp(gamma*x)*x)/np.sum(np.exp(gamma*x))


'''
Euler forward integration of the dynamics under the policy
在给定的动态模型 dyn 和时间步长 ts 的情况下，使用欧拉前向积分方法对状态 x 进行 N 步的预测。
输入:
    x: 初始状态(x,y,v,ψ);   dyn: 动态模型函数,输入状态x和控制输入u,输出状态导数xdot.
    N: 预测步数;   ts: 时间步长.
输出:
    xs: 预测N步的状态列表 xs.
'''
def propagate_backup(x,dyn,N,ts):
    # 根据状态 x 的类型定义预测的N步的状态列表 xs 的类型
    if isinstance(x,numpy.ndarray):
        xs = np.empty([N,x.shape[0]])
    elif isinstance(x,casadi.SX):
        xs = SX(N,x.shape[0])
    elif isinstance(x,casadi.MX):
        xs = MX(N,x.shape[0])
    # xs = np.zeros(N,x.shape[0])

    for i in range(0,N):
        x = x+dyn(x)*ts
        xs[i,:]=x
    return xs

# 计算状态x是与道路边界是否碰撞,h>0为无碰撞。(即状态x的y坐标是否在道路边界lb~ub之间为安全.)
def lane_bdry_h(x, lb=0, ub=7.2):
    if isinstance(x,casadi.SX):
        h = SX(x.shape[0],1)
        for i in range(0,x.shape[0]):
            h[i] = softmin(vertcat(x[i,1]-lb,ub-x[i,1]),5)
        return h
    elif isinstance(x,casadi.MX):
        h = MX(x.shape[0],1)
        for i in range(0,x.shape[0]):
            h[i] = softmin(vertcat(x[i,1]-lb,ub-x[i,1]),5)
        return h
    else:
        if x.ndim==1:
            return softmin(np.array([x[1]-lb,ub-x[1]]),5)
        else:
            h = np.zeros(x.shape[0])
            for i in range(0,x.shape[0]):
                h[i] = softmin(np.array([x[i,1]-lb,ub-x[i,1]]),5)
            return h

# 计算 ego 当前状态 x1 与 obs 当前状态 x2 的碰撞安全值 h 并返回。 (当 h>0 时表示无碰撞)
def veh_col(x1,x2,size,alpha=1):
    '''
    vehicle collision constraints: h>=0 means no collision
    implemented via a softmax function
    '''
    if isinstance(x1,casadi.SX):
        h = SX(x1.shape[0])
        for i in range(0,x1.shape[0]):
            dx = (fabs(x1[i,0]-x2[i,0])-size[0])
            dy = (fabs(x1[i,1]-x2[i,1])-size[1])
            h[i] = (dx*np.exp(alpha*dx)+dy*np.exp(dy*alpha))/(np.exp(alpha*dx)+np.exp(dy*alpha))
        return h
    elif isinstance(x1,casadi.MX):
        h = MX(x1.shape[0])
        for i in range(0,x1.shape[0]):
            dx = (fabs(x1[i,0]-x2[i,0])-size[0])
            dy = (fabs(x1[i,1]-x2[i,1])-size[1])
            h[i] = (dx*np.exp(alpha*dx)+dy*np.exp(dy*alpha))/(np.exp(alpha*dx)+np.exp(dy*alpha))
        return h
    else:
        if x1.ndim==1:
            dx = np.clip((abs(x1[0]-x2[0])-size[0]),-5,5)
            dy = np.clip((abs(x1[1]-x2[1])-size[1]),-5,5)
            return (dx*np.exp(alpha*dx)+dy*np.exp(dy*alpha))/(np.exp(alpha*dx)+np.exp(dy*alpha))
        else:
            h = np.zeros(x1.shape[0])
            for i in range(0,x1.shape[0]):
                dx = np.clip((abs(x1[i][0]-x2[i][0])-size[0]),-5,5)
                dy = np.clip((abs(x1[i][1]-x2[i][1])-size[1]),-5,5)
                h[i] = (dx*np.exp(alpha*dx)+dy*np.exp(dy*alpha))/(np.exp(alpha*dx)+np.exp(dy*alpha))
            return h


class PredictiveModel():
    # N_lane: road上的车道数量
    def __init__(self, n, d, N, backupcons, dt, cons,N_lane = 3):
        self.n = n # state dimension
        self.d = d # input dimention
        self.N = N # number of prediction steps
        self.m = len(backupcons) # number of policies(3个:maintain,brake,lc)
        self.dt = dt
        self.cons = cons #parameters
        self.Asym = None    # 状态矩阵A
        self.Bsym = None    # 输入矩阵B
        self.xpsym = None   # 状态预测函数
        self.hsym = None    # 碰撞检测函数
        self.zpred = None   # 备份策略预测
        self.xpred = None   # 主策略预测
        self.u0sym = None   # 初始控制输入
        self.Jh = None      # 碰撞检测的雅可比矩阵
        self.LB = [cons.W/2, N_lane*3.6-cons.W/2] #lane boundary(是road边界)
        self.backupcons = backupcons    # 备份策略
        self.calc_xp_expr() # 计算表达式


    def dyn_linearization(self, x,u):
        #linearizing the dynamics x^+=Ax+Bu+C
        A = self.Asym(x,u)
        B = self.Bsym(x,u)
        xp = self.xpsym(x,u)
        C = xp-A@x-B@u

        return np.array(A),np.array(B),np.squeeze(np.array(C)),np.squeeze(np.array(xp))

    def branch_eval(self,x,z):
        p = self.psym(x,z)
        dp = self.dpsym(x,z)
        return np.array(p).flatten(),np.array(dp)
    def zpred_eval(self,z):
        return np.array(self.zpred(z))
    def xpred_eval(self,x):
        return self.xpred(x),self.u0sym(x)
    def col_eval(self,x,z):
        dh = np.squeeze(np.array(self.dhsym(x,z)))
        h = np.squeeze(np.array(self.hsym(x,z)))
        return h-np.dot(dh,x),dh

    def update_backup(self,backupcons):
        self.backupcons = backupcons
        self.m = len(backupcons)
        self.calc_xp_expr()

    """ 根据预测范围N内obs与ego是否有碰撞, 以及obs与道路边界是否碰撞, 计算碰撞结果h.(h>0为无碰撞)
    """
    def BF_traj(self,x1,x2):
        if isinstance(x1,casadi.SX):
            h = SX(x1.shape[0]*2,1)
        elif isinstance(x1,casadi.MX):
            h = MX(x1.shape[0]*2,1)

        # h大小为(2*N, 1),前0~N为ego预测轨迹与obs预测轨迹的碰撞结果,后N~2N为ego预测轨迹与道路边界的碰撞结果
        for i in range(0,x1.shape[0]):
            # 第i个预测步obs状态x1[i,:]与ego状态x2[i,:]的碰撞结果。(h[i]>0意味着无碰撞)
            h[i] = veh_col(x1[i,:], x2[i,:], [self.cons.L + 2,self.cons.W + 0.2]) 
            # 第i步obs预测状态是否与道路边界是否碰撞。(h>0为无碰撞)
            h[i+x1.shape[0]] = lane_bdry_h(x1[i,:], self.LB[0], self.LB[1])
        return softmin(h,5)

    # 根据每个策略的安全性h,计算每种策略的分支概率p. (返回概率结果的维度与 h 相同)
    def branch_prob(self,h):
        #branching probablity as a function of the safety function
        h = softsat(h,1)        # softsat 函数对所有策略的安全性数组 h 进行平滑处理，避免概率突变
        m = exp(self.cons.s1*h) # 使用指数函数对平滑后的安全值 h 进行加权
        return m/sum1(m)        # 对加权结果进行归一化，得到每种备份策略的分支概率。


    """ 计算和构建预测模型所需的表达式。
    """
    def calc_xp_expr(self):
        u = SX.sym('u', self.d) # 控制输入符号变量
        x = SX.sym('x', self.n)  # ego 状态符号变量
        z = SX.sym('z', self.n)  # obs 状态符号变量
        zdot = SX.sym('zdot', self.m, self.n)

        xp = x + dubin(x,u)*self.dt   # ego 状态预测(状态转移方程)

        dyn = lambda x:dubin(x, self.backupcons[0](x))  #计算状态x在策略maintain下的状态导数(即定义maintain下的动态模型函数)
        x1 = propagate_backup(x, dyn, self.N, self.dt)  # ego 在 maintain 策略下的N步预测轨迹(时间步长ts).
        # x2 记录 obs 在self.m中策略下的预测轨迹。 x2是2维的,第1维size为self.N是预测步数,第2维size为self.m*self.n,每间隔n列记录一个策略下的预测轨迹,因此共n*m列。
        x2 = SX(self.N, self.n*self.m)
        for i in range(0,self.m):
            dyn = lambda x:dubin(x,self.backupcons[i](x))   # 第i个策略下的动态模型函数
            x2[:, i*self.n:(i+1)*self.n] = propagate_backup(z, dyn, self.N, self.dt)    # obs在第i个策略下的预测轨迹记录在x2的第(i*n ~ i*(n+1))列中
        # hi 是每个策略的安全性: obs 在m个策略下的预测轨迹 x2 与 ego 的预测轨迹 x1 之间的碰撞结果; 以及 obs 与 road 边界的碰撞结果。
        hi = SX(self.m,1)   # hi大小为(m*1), 每一行对应obs在该策略的碰撞结果。
        for i in range(0,self.m):
            # 如果h[i]>0,则表示obs在第i个策略下的预测轨迹安全无碰撞。
            hi[i] = self.BF_traj(x2[:, self.n*i:self.n*(i+1)], x1)
        p = self.branch_prob(hi)    # 根据每个策略的安全性h,计算每种策略的分支概率p

        # 计算 ego(x) 与 obs(z) 的碰撞安全值 h(h>0意味着无碰撞)
        h = veh_col(x.T, z.T, [self.cons.L+1, self.cons.W+0.2], 1)

        # Function() 是 CasADi 库函数,用于定义符号函数.基本语法: Function(name, inputs, outputs)
        self.xpsym = Function('xp',[x,u],[xp])      # 系统状态转移方程，输入x,u,输出下一个状态xp
        self.Asym = Function('A',[x,u],[jacobian(xp,x)])    # 求状态转移方程的A矩阵
        self.Bsym = Function('B',[x,u],[jacobian(xp,u)])    # 求状态转移方程的B矩阵
        self.dhsym = Function('dh',[x,z],[jacobian(h,x)])   # ego与obs当前状态的碰撞检测h对x的偏导
        self.hsym = Function('h',[x,z],[h])     # ego与obs当前状态的碰撞检测函数
        self.zpred = Function('zpred',[z],[x2]) #
        self.xpred = Function('xpred',[x],[x1]) #
        self.psym = Function('p',[x,z],[p])     #
        self.dpsym = Function('dp',[x,z],[jacobian(p,x)])       #
        self.u0sym = Function('u0',[x],[self.backupcons[0](x)]) #

class PredictiveModel_merge():
    def __init__(self, n, d, N, backupcons, dt, cons, merge_ref, laneID = 0, N_lane1 = 3, N_lane2 = 2):
        '''
        Similar object, this one for the merging case. Because lookup table is needed, we use CasADi.MX variable instead of SX, which is slower.
        '''
        self.n = n # state dimension
        self.d = d # input dimention
        self.N = N # number of prediction steps
        self.m = len(backupcons)
        self.dt = dt
        self.cons = cons
        self.Asym = None
        self.Bsym = None
        self.xpsym = None
        self.hsym = None
        self.zpred = None
        self.xpred = None
        self.u0sym = None
        self.Jh = None
        self.N_lane2 = N_lane2
        self.laneID = laneID
        self.refY = merge_ref[0]
        self.refpsi = merge_ref[1]
        self.LB1 = [cons.W/2,N_lane1*3.6-cons.W/2]

        self.backupcons = backupcons

        self.calc_xp_expr()



    def dyn_linearization(self, x,u):

        A = self.Asym(x,u)
        B = self.Bsym(x,u)
        xp = self.xpsym(x,u)
        C = xp-A@x-B@u

        return np.array(A),np.array(B),np.squeeze(np.array(C)),np.squeeze(np.array(xp))

    def branch_eval(self,x,z):
        p = self.psym(x,z)
        dp = self.dpsym(x,z)
        return np.array(p).flatten(),np.array(dp)
    def zpred_eval(self,z):
        return np.array(self.zpred(z))
    def xpred_eval(self,x):
        return self.xpred(x),self.u0sym(x)
    def col_eval(self,x,z):
        dh = np.squeeze(np.array(self.dhsym(x,z)))
        h = np.squeeze(np.array(self.hsym(x,z)))
        return h-np.dot(dh,x),dh


    def update_backup(self,backupcons):
        self.backupcons = backupcons
        self.m = len(backupcons)
        self.calc_xp_expr()



    def BF_traj(self,x1,x2):
        h = MX(x1.shape[0],1)
        for i in range(0,x1.shape[0]):
            h[i] = veh_col(x1[i,:],x2[i,:],[self.cons.L+1,self.cons.W+0.2])
        return softmin(h,5)
    def branch_prob(self,h):
        h = softsat(h,1)
        m = exp(self.cons.s1*h)
        return m/sum1(m)

    def calc_xp_expr(self):
        u = MX.sym('u', self.d)
        x = MX.sym('x',self.n)
        z = MX.sym('z',self.n)
        zdot = MX.sym('zdot',self.m,self.n)

        xp = x+dubin(x,u)*self.dt

        dyn = lambda x:dubin(x,self.backupcons[0](x))
        x1 = propagate_backup(x,dyn,self.N,self.dt)
        x2 = MX(self.N,self.n*self.m)
        for i in range(0,self.m):
            dyn = lambda x:dubin(x,self.backupcons[i](x))
            x2[:,i*self.n:(i+1)*self.n] = propagate_backup(z,dyn,self.N,self.dt)
        hi = MX(self.m,1)
        for i in range(0,self.m):
            hi[i] = self.BF_traj(x2[:,self.n*i:self.n*(i+1)],x1)
        p = self.branch_prob(hi)

        h = veh_col(x.T,z.T,[self.cons.L+1,self.cons.W+0.2],1)

        self.xpsym = Function('xp',[x,u],[xp])
        self.Asym = Function('A',[x,u],[jacobian(xp,x)])
        self.Bsym = Function('B',[x,u],[jacobian(xp,u)])
        self.dhsym = Function('dh',[x,z],[jacobian(h,x)])
        self.hsym = Function('h',[x,z],[h])
        self.zpred = Function('zpred',[z],[x2])
        self.xpred = Function('xpred',[x],[x1])
        self.psym = Function('p',[x,z],[p])
        self.dpsym = Function('dp',[x,z],[jacobian(p,x)])
        self.u0sym = Function('u0',[x],[self.backupcons[0](x)])
        self.hisym = Function('hi',[x,z],[hi])
