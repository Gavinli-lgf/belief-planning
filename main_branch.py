import sys
from Init_MPC import initBranchMPC
from MPC_branch import BranchMPC, robustMPC, BranchMPC_CVaR, BranchMPCParams
from highway_branch_dyn import *
import numpy as np
import pdb
import Highway_env_branch
from Highway_env_branch import merge_geometry
from utils import Branch_constants, MPCParams

"""
功能: overtake之后进行lane change
1. 先初始化地图,车辆,mpc的一些基础参数;
2. 在初始化lane change的目标状态,策略列表;
3. 定义Predictive model对象 model,用于生成scenario tree;
4. 定义Branch mpc的初始参数对象 mpcParam,该对象存储了从scenario tree生成trajectory tree过程所需参数;
5. 根据mode,mpcParam,并引入CVaR,定义branch mpc cvar对象 mpc,可实现从scenario tree生成trajectory tree的整个求解过程;
6. 最后将 mpc 对象,N_lane=4代入仿真环境Highway_env_branch中,进行整体求解与仿真。
"""
def sim_overtake():
    # ======================================================================================================================
    # ============================================= Initialize parameters  =================================================
    # ======================================================================================================================
    N = 8                                # number of time steps for each branch(对应论文中，一个branch的仿真步数)
    n = 4;   d = 2                       # State and Input dimension(状态(x,y,v,ψ);输入(a,r))
    x0 = np.array([0, 1.8, 0, 0])        # Initial condition (only for initializing the MPC, not the actual initial state of the sim)
    am = 6.0                             # maximum acceleration       
    rm = 0.3                             # maximum steering angle rate
    dt = 0.1
    NB = 2                               # number of branching, 2 means a tree with 1-m-m^2 branches at each level.(对应论文中分支节点的子节点数)

    N_lane = 4

    # Initialize controller parameters(先初始化model与mpcParam；然后由mpcParam,model,ralpha=0.1生成mpc；最后将mpc,N_lane=4代入Highway_env_branch进行整体求解与仿真。)
    xRef = np.array([0.5, 1.8, 15, 0]) # 要lane change时的目标状态
    # Branch_constants:定义了分支预测,碰撞检测,车辆运动学模型中的必要的配置参数
    cons =  Branch_constants(s1=2,s2=3,c2=0.5,tran_diag=0.3,alpha=1,R=1.2,am=am,rm=rm,J_c=20,s_c=1,ylb = 0.,yub = 7.2,L=4, W=2.5,col_alpha=5,Kpsi = 0.1)
    # 包含三种控制策略lambda函数的列表(maintain,brake,lc)
    backupcons = [lambda x:backup_maintain(x,cons), lambda x:backup_brake(x,cons), lambda x:backup_lc(x,xRef)]
    # Predictive model and scenario tree
    model = PredictiveModel(n, d, N, backupcons, dt, cons)

    #定义了"cost function,state constraints,以及其他使用mpc方法由scenario tree生成trajectory tree"相关的参数.
    mpcParam = initBranchMPC(n,d,N,NB,xRef,am,rm,N_lane,cons.W)
    # mpc = robustMPC(mpcParam, model)
    # mpc = BranchMPC(mpcParam, model)
    #定义了有CVaR的目标函数(对应论文中:"IV. BRANCH MPC WITH RISK MEASURE OBJECTIVES")
    mpc = BranchMPC_CVaR(mpcParam, model, ralpha=0.1)

    # 实际运行过程中就只需调用mpc(有CVaR的目标函数)求解即可。(model,mpcParam等只是mpc对象的一部分)
    Highway_env_branch.sim_overtake(mpc,N_lane)

def sim_merge():


    # Initialize controller parameters

    lane_width = 3.6

    N = 40
    n = 4;   d = 2                            # State and Input dimension
    x0 = np.array([0, 1.8, 0, 0])        # Initial condition
    xRef = np.array([0.5,1.8,15,0])
    am = 7.0
    rm = 0.3
    dt = 0.1
    NB = 1
    N_lane = 2
    cons =  Branch_constants(s1=2,s2=3,c2=0.5,tran_diag=0.3,alpha=1,R=1.2,am=am,rm=rm,J_c=20,s_c=1,ylb = 0.,yub = 7.2,L=4, W=2.5,col_alpha=5,Kpsi = 0.1)
    merge_lane=1
    merge_s = 50
    merge_R=300
    merge_side = 0
    merge_lane_ref_X1,merge_lane_ref_X2,merge_lane_ref_Y1,merge_lane_ref_Y2,merge_lane_ref_psi1,merge_lane_ref_psi2 = merge_geometry(N_lane,merge_lane,merge_s,merge_R, merge_side)
    merge_lane_ref_Y = np.append(merge_lane_ref_Y1,merge_lane_ref_Y2)
    merge_lane_ref_X = np.append(merge_lane_ref_X1,merge_lane_ref_X2)
    merge_lane_ref_psi = np.append(merge_lane_ref_psi1,merge_lane_ref_psi2)
    refY = interpolant('refY','linear',[merge_lane_ref_X],merge_lane_ref_Y)
    refpsi = interpolant('refpsi','linear',[merge_lane_ref_X],merge_lane_ref_psi)
    merge_ref = (refY,refpsi)
    from Highway_env_branch import v0
    backupcons_merge = [lambda x:backup_maintain_trackV(x,cons,v0,refpsi),lambda x:backup_brake(x,cons,refpsi)]
    backupcons_normal = [lambda x:backup_maintain_trackV(x,cons,v0),lambda x:backup_brake(x,cons)]
    pred_model = [PredictiveModel_merge(n, d, N, backupcons_normal, dt, cons, merge_ref, laneID = 0, N_lane1 = N_lane, N_lane2 = merge_lane),\
                  PredictiveModel_merge(n, d, N, backupcons_merge, dt, cons, merge_ref, laneID = 1, N_lane1 = N_lane, N_lane2 = merge_lane)]
    mpcParam = initBranchMPC(n,d,N,NB,xRef,am,rm,N_lane,cons.W)
    mpc = BranchMPC_CVaR(mpcParam, pred_model[0],ralpha=0.1)
    Highway_env_branch.sim_merge(mpc,pred_model,N_lane,merge_lane,merge_s,merge_R,merge_side)


if __name__== "__main__":
  sim_overtake()
#   sim_merge()
