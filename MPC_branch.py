import pdb
import numpy as np
from cvxopt import spmatrix, matrix, solvers
from numpy import linalg as la
from scipy import linalg
from scipy import sparse
from cvxopt.solvers import qp
import datetime
from numpy import hstack, inf, ones
from scipy.sparse import vstack
from osqp import OSQP
from dataclasses import dataclass, field
import ecos

solvers.options['show_progress'] = False

@dataclass
class PythonMsg:
    def __setattr__(self,key,value):
        if not hasattr(self,key):
            raise TypeError ('Cannot add new field "%s" to frozen class %s' %(key,self))
        else:
            object.__setattr__(self,key,value)

# BranchMPCParams 类用于存储和管理分支MPC控制器的各种参数和配置。
@dataclass
class BranchMPCParams(PythonMsg):
    n: int = field(default=None) # dimension state space
    d: int = field(default=None) # dimension input space
    NB: int = field(default=None) # number of branching
    N: int = field(default=None)      # number of time steps in between branches

    A: np.array = field(default=None) # prediction matrices. Single matrix for LTI and list for LTV
    B: np.array = field(default=None) # prediction matrices. Single matrix for LTI and list for LTV

    Q: np.array = field(default=np.array((n, n))) # quadratic state cost
    R: np.array = field(default=None) # quadratic input cost
    Qf: np.array = field(default=None) # quadratic state cost final
    dR: np.array = field(default=None) # Quadratic rate cost

    Qslack: float = field(default=None) # it has to be a vector. Qslack = [linearSlackCost, quadraticSlackCost]
    Fx: np.array = field(default=None) # State constraint Fx * x <= bx(每个状态节点都有约束,Fx是(4*4),bx是(4*1))
    bx: np.array = field(default=None)
    Fu: np.array = field(default=None) # State constraint Fu * u <= bu(Fu是(4*2),bu是(4*1))
    bu: np.array = field(default=None)
    xRef: np.array = field(default=None)

    slacks: bool = field(default=True)
    timeVarying: bool = field(default=False)

    def __post_init__(self):
        if self.Qf is None: self.Qf = self.Q
        if self.dR is None: self.dR = np.zeros(self.d)
        if self.xRef is None: self.xRef = np.zeros(self.n)

############################################################################################
####################################### MPC CLASS ##########################################
############################################################################################

"""
功能: 定义Branch结构(场景树与轨迹树branch结构相同,包含N个时间步)
输入: xtraj:(N*n),ego的branch; ztraj:(N*n),obs的branch; utraj:(n*d),ego输入预测; w:branch的概率(权重), depth:branch的深度(处于第几层)
     (注branch的大小N不完全相同:根branch的N=1; 其它branch的N=8.)
"""
class BranchTree():
    def __init__(self,xtraj,ztraj,utraj,w,depth=0):
        self.xtraj = xtraj  # 该branch的 ego 预测(N*n)
        self.ztraj = ztraj  # 该branch的 obs 预测(N*n)
        self.utraj = utraj  # 该branch的 ego 输入(n*d)
        self.dynmatr = [None]*xtraj.shape[0]    #每个时间步ego的动态模型参数A,B,C(每个时间步的x,u不同,则线性化后的结果也不同)
        self.w = w          # 该Branch的概率(权重)
        self.children = []  # 该Branch的子Branch列表(个数就是策略个数m)
        self.depth = depth  # 该Branch的深度
        self.p = None       # 存储该Branch的子branch的概率(m*1)
        self.dp = None      # 上面p对ego转态x的导数(m*1)
        self.J = 0
    def addchild(self,BT):
        self.children.append(BT)



class BranchMPCProx():

    def __init__(self,  mpcParameters, predictiveModel):
        """Initialization
        Arguments:
            mpcParameters: struct containing MPC parameters
            predictiveModel: containing CasADi functions about z prediction, collision constraints, and branching probability
        """
        self.N      = mpcParameters.N
        self.NB     = mpcParameters.NB
        self.Qslack = mpcParameters.Qslack
        self.Q      = mpcParameters.Q
        self.Qf     = mpcParameters.Qf
        self.R      = mpcParameters.R
        self.dR     = mpcParameters.dR
        self.n      = mpcParameters.n
        self.d      = mpcParameters.d
        self.Fx     = mpcParameters.Fx
        self.Fu     = mpcParameters.Fu
        self.bx     = mpcParameters.bx
        self.bu     = mpcParameters.bu
        self.xRef   = mpcParameters.xRef
        self.m      = predictiveModel.m

        self.slacks          = mpcParameters.slacks
        self.slackweight     = None
        self.timeVarying     = mpcParameters.timeVarying
        self.predictiveModel = predictiveModel
        self.osqp = None
        self.BT = None
        self.totalx = 0
        self.totalu = 0
        self.ndx = {}
        self.ndu = {}

        self.xPred = None
        self.uPred = None
        self.xLin = None
        self.uLin = None
        self.OldInput = np.zeros(self.d)

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.timeStep = 0
    def inittree(self,x,z):
        '''
        Initialize the scenario tree, each node contains the z prediction, x prediction (from last iteration) and u prediction (from last iteration), and weight
        '''
        u = np.zeros(self.d)
        self.BT = BranchTree(np.reshape(x,[1,self.n]),np.reshape(z,[1,self.n]),np.reshape(u,[1,self.d]),1,0)
        q = [self.BT]
        countx = 0
        countu = 0
        self.uLin = np.reshape(u,[1,self.d])
        self.xLin = np.reshape(x,[1,self.n])

        self.ndx[self.BT] = countx
        self.ndu[self.BT] = countu
        A,B,C,xp = self.predictiveModel.dyn_linearization(x,u)
        self.BT.dynmatr[0] = (A,B,C)
        countx+=self.BT.xtraj.shape[0]
        countu+=self.BT.xtraj.shape[0]

        while len(q)>0:
            currentbranch = q.pop(0)

            if currentbranch.depth<self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1],currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp= dp
                for i in range(0,self.m):
                    xtraj = np.zeros((self.N,self.n))
                    utraj = np.zeros((self.N,self.d))
                    newbranch = BranchTree(xtraj,zPred[:,self.n*i:self.n*(i+1)],utraj,p[i]*currentbranch.w,currentbranch.depth+1)
                    A,B,C,xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],currentbranch.utraj[-1])
                    newbranch.xtraj[0] = xp
                    for t in range(0,self.N):
                        A,B,C,xp = self.predictiveModel.dyn_linearization(newbranch.xtraj[t],newbranch.utraj[t])
                        newbranch.dynmatr[t] = (A,B,C)
                        if t<self.N-1:
                            newbranch.xtraj[t+1] = xp

                    self.ndx[newbranch] = countx
                    self.ndu[newbranch] = countu

                    self.xLin = np.vstack((self.xLin,newbranch.xtraj))
                    self.uLin = np.vstack((self.uLin,newbranch.utraj))
                    if newbranch.depth == self.NB:
                        countx+=(newbranch.xtraj.shape[0]+1)
                    else:
                        countx+=newbranch.xtraj.shape[0]
                    countu+=newbranch.xtraj.shape[0]
                    currentbranch.addchild(newbranch)
                    q.append(newbranch)
        self.totalx = countx
        self.totalu = countu
        self.slackweight = np.zeros(self.totalx*(self.Fx.shape[0]+1))


    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.totalx*self.n)
        Gu = np.zeros((self.totalx*self.n, self.totalu*self.d))

        E = np.zeros((self.totalx*self.n, self.n))
        E[0:self.n] = np.eye(self.n)

        L = np.zeros(self.totalx*self.n)
        self.E = E

        totalxdim = self.totalx*self.n
        for branch in self.ndx:
            l = branch.xtraj.shape[0]
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            for t in range(1,l):
                A,B,C = branch.dynmatr[t-1]
                Gx[(ndx+t)*self.n:(ndx+t+1)*self.n,(ndx+t-1)*self.n:(ndx+t)*self.n] = -A
                Gu[(ndx+t)*self.n:(ndx+t+1)*self.n,(ndu+t-1)*self.d:(ndu+t)*self.d] = -B
                L[(ndx+t)*self.n:(ndx+t+1)*self.n]                                  = C
            A,B,C = branch.dynmatr[-1]
            if branch.depth<self.NB:
                for child in branch.children:
                    ndxc = self.ndx[child]
                    Gx[ndxc*self.n:(ndxc+1)*self.n,(ndx+l-1)*self.n:(ndx+l)*self.n] = -A
                    Gu[ndxc*self.n:(ndxc+1)*self.n,(ndu+l-1)*self.d:(ndu+l)*self.d] = -B
                    L[ndxc*self.n:(ndxc+1)*self.n]                                  = C
            else:
                Gx[(ndx+l)*self.n:(ndx+l+1)*self.n,(ndx+l-1)*self.n:(ndx+l)*self.n] = -A
                Gu[(ndx+l)*self.n:(ndx+l+1)*self.n,(ndu+l-1)*self.d:(ndu+l)*self.d] = -B
                L[(ndx+l)*self.n:(ndx+l+1)*self.n]                                  = C
        self.L = L

        if self.slacks == True:
            self.G = np.hstack( (Gx, Gu, np.zeros( ( Gx.shape[0], self.slackweight.shape[0]) ) ) )
        else:
            self.G = np.hstack((Gx, Gu))

    def updatetree(self,x,z):
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            branch.utraj[0:l-1] = self.uLin[self.ndu[branch]+1:self.ndu[branch]+l]
            if branch.depth<self.NB:
                idx = np.argmax(branch.p)
                ndu = self.ndu[branch.children[idx]]
                branch.utraj[-1] = self.uLin[ndu]
            else:
                branch.utraj[-1] = branch.utraj[-2]
        self.BT.ztraj = np.reshape(z,[1,self.n])
        self.BT.xtraj = np.reshape(x,[1,self.n])
        for i in range(0,self.BT.xtraj.shape[0]):
            A,B,C,xp = self.predictiveModel.dyn_linearization(self.BT.xtraj[i],self.BT.utraj[i])
            self.BT.dynmatr[i]=(A,B,C)
        q = [self.BT]

        while len(q)>0:
            currentbranch = q.pop(0)
            if currentbranch.depth<self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1],currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp = dp
                for i in range(0,self.m):
                    child = currentbranch.children[i]
                    child.w = currentbranch.w*p[i]
                    child.ztraj = zPred[:,i*self.n:(i+1)*self.n]
                    xtraj = np.zeros((self.N,self.n))
                    A,B,C,xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],currentbranch.utraj[-1])
                    child.xtraj[0] = xp
                    for t in range(0,self.N):
                        A,B,C,xp = self.predictiveModel.dyn_linearization(child.xtraj[t],child.utraj[t])
                        child.dynmatr[t] = (A,B,C)
                        if t<self.N-1:
                            child.xtraj[t+1] = xp

                    q.append(child)


    def buildCost(self):
        totalxdim = self.totalx*self.n
        listQ = [None] * (self.totalx)
        Hu = np.zeros([self.totalu*self.d,self.totalu*self.d])
        dRmat = np.diag(self.dR)
        qx = np.zeros(self.totalx*self.n)
        dQ = self.Q*3
        for branch in self.ndx:
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            l = branch.utraj.shape[0]
            for i in range(0,l-1):
                # t = 1+self.N*(branch.depth-1)+i
                listQ[ndx+i]=(dQ+self.Q)*branch.w
                qx[(ndx+i)*self.n:(ndx+i+1)*self.n] = -2*branch.w*(np.dot(self.xRef,self.Q)+np.dot(branch.xtraj[i],dQ))
                Hu[(ndu+i)*self.d:(ndu+i+1)*self.d,(ndu+i)*self.d:(ndu+i+1)*self.d] += branch.w*self.R
                Hu[(ndu+i)*self.d:(ndu+i+1)*self.d,(ndu+i)*self.d:(ndu+i+1)*self.d] += branch.w*dRmat
                Hu[(ndu+i)*self.d:(ndu+i+1)*self.d,(ndu+i+1)*self.d:(ndu+i+2)*self.d] -= branch.w*dRmat
                Hu[(ndu+i+1)*self.d:(ndu+i+2)*self.d,(ndu+i)*self.d:(ndu+i+1)*self.d] -= branch.w*dRmat
                Hu[(ndu+i+1)*self.d:(ndu+i+2)*self.d,(ndu+i+1)*self.d:(ndu+i+2)*self.d] +=branch.w*dRmat


            if branch.depth<self.NB:
                Hu[(ndu+l-1)*self.d:(ndu+l)*self.d,(ndu+l-1)*self.d:(ndu+l)*self.d] += branch.w*(self.R+dRmat)

                listQ[ndx+l-1] = (dQ+self.Q)*branch.w
                childJ = np.zeros(self.m)
                for j in range(0,self.m):
                    childJ[j] = branch.children[j].J
                    ndu_child = self.ndu[branch.children[j]]
                    Hu[(ndu+l-1)*self.d:(ndu+l)*self.d,(ndu_child)*self.d:(ndu_child+1)*self.d] -= branch.children[j].w*dRmat
                    Hu[(ndu_child)*self.d:(ndu_child+1)*self.d,(ndu+l-1)*self.d:(ndu+l)*self.d] -= branch.children[j].w*dRmat
                    Hu[(ndu_child)*self.d:(ndu_child+1)*self.d,(ndu_child)*self.d:(ndu_child+1)*self.d] += branch.children[j].w*dRmat

                qx[(ndx+l-1)*self.n:(ndx+l)*self.n] = branch.w*(-2*np.dot(self.xRef,self.Q)-2*np.dot(branch.xtraj[-1],dQ)+np.dot(childJ,branch.dp))


            else:
                Hu[(ndu+l-1)*self.d:(ndu+l)*self.d,(ndu+l-1)*self.d:(ndu+l)*self.d] = branch.w*self.R
                listQ[ndx+l-1] = (dQ+self.Q)*branch.w
                listQ[ndx+l] = self.Qf*branch.w

                qx[(ndx+l-1)*self.n:(ndx+l)*self.n] = -2*branch.w*(np.dot(self.xRef,self.Q)+np.dot(branch.xtraj[l-1],dQ))
                qx[(ndx+l)*self.n:(ndx+l+1)*self.n] = -2*branch.w*(np.dot(self.xRef,self.Qf))
        Hx = linalg.block_diag(*listQ)
        qu = np.zeros(self.totalu*self.d)
        qu[0:self.d] = -2*self.OldInput @ self.dR
        Hu[0:self.d,0:self.d]+=self.dR

        # Cost linear term for state and input
        q = np.append(qx,qu)

        if self.slacks == True:
            quadSlack = self.Qslack[0] * np.eye(self.slackweight.shape[0])
            linSlack  = self.Qslack[1] * self.slackweight
            self.H = linalg.block_diag(Hx, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else:
            self.H = linalg.block_diag(Hx, Hu)
            self.q = q
        self.H = 2*self.H  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b
        # Let's start by computing the submatrix of F relates with the state
        Nc = self.Fx.shape[0]+1
        slackweight_x = np.zeros(self.totalx*Nc)


        Fxtot = np.zeros([Nc*self.totalx,self.totalx*self.n])
        bxtot = np.zeros(Nc*self.totalx)
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0,l):
                # collision constraint linearized
                h,dh = self.predictiveModel.col_eval(branch.xtraj[i],branch.ztraj[i])
                idx = self.ndx[branch]+i
                Fxtot[idx*Nc:(idx+1)*Nc,idx*self.n:(idx+1)*self.n] = np.vstack((-dh,self.Fx))
                bxtot[idx*Nc:(idx+1)*Nc] = np.append(h,self.bx)
                slackweight_x[idx*Nc:(idx+1)*Nc] = branch.w

        self.slackweight = slackweight_x
        rep_b = [self.Fu] * (self.totalu)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.totalu)

        # Let's stack all together
        F_hard = linalg.block_diag(Fxtot, Futot)

        # Add slack if need

        if self.slacks == True:
            nc_x = Fxtot.shape[0] # add slack only for state constraints
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc_x))
            addSlack[0:nc_x, 0:nc_x] = -np.eye(nc_x)
            # Now constraint slacks >= 0
            I = - np.eye(nc_x); Zeros = np.zeros((nc_x, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = np.vstack(( np.hstack((F_hard, addSlack)) , Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x)))
        else:
            self.F = F_hard
            self.b = np.hstack((bxtot, butot))
    def updateIneqConstr(self):
        # for warm_start, faster than building the constraints from scratch
        Nc = self.Fx.shape[0]+1
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0,l):
                h,dh = self.predictiveModel.col_eval(branch.xtraj[i],branch.ztraj[i])
                idx = self.ndx[branch]+i
                self.F[idx*Nc,idx*self.n:(idx+1)*self.n] = -dh
                self.b[idx*Nc] = h
                self.slackweight[idx*Nc:(idx+1)*Nc] = branch.w


    def solve(self, x,z,xRef=None):
        """Computes control action
        Arguments:
            x: ego vehicle state
            z: uncontrolled vehicle state
        """

        if not xRef is None:
            self.xRef = xRef
        if self.BT is None:
            self.inittree(x,z)
            self.buildIneqConstr()
        else:
            self.updatetree(x,z)
            self.updateIneqConstr()

        self.buildCost()
        self.buildEqConstr()


        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L
        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,x),self.L_FTOCP))
        self.unpackSolution()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        # print("Solver Time: ", self.solverTime.total_seconds(), " seconds.")


        # update applied input
        self.OldInput = self.uPred[0,:]
        self.timeStep += 1


    def addTerminalComponents(self):
        # TO DO: ....
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L

    def unpackSolution(self):
        # Extract predicted state and predicted input trajectories
        if self.feasible:
            self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.totalx*self.n)]),(-1,self.n)))).T
            self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.totalx*self.n+np.arange(self.totalu*self.d)]),(-1, self.d)))).T
            self.xLin = self.xPred
            self.uLin = self.uPred
            self.uLin = np.vstack((self.uLin,self.uLin[-1]))

    def BT2array(self):
        # for plotting and dubug maining, outputs the scenario tree as x and z trajectories
        ztraj = []
        xtraj = []
        utraj = []
        branch_w = []
        q = [self.BT]
        while (len(q)>0):
            curr = q.pop(0)
            for child in curr.children:
                branch_w.append(child.w)
                ztraj.append(np.vstack((curr.ztraj[-1],child.ztraj)))
                xtraj.append(np.vstack((curr.xtraj[-1],child.xtraj)))
                utraj.append(np.vstack((curr.utraj[-1],child.utraj)))
                q.append(child)
        return xtraj,ztraj,utraj,branch_w

    def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """
        qp_A = vstack([G, A]).tocsc()
        l = -inf * ones(len(h))
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])

        self.osqp = OSQP()

        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0

        self.Solution = res.x


class BranchMPC():

    def __init__(self,  mpcParameters, predictiveModel):
        """Initialization
        Arguments:
            mpcParameters: struct containing MPC parameters
            predictiveModel: containing CasADi functions about z prediction, collision constraints, and branching probability
        """
        self.N      = mpcParameters.N
        self.NB     = mpcParameters.NB
        self.Qslack = mpcParameters.Qslack
        self.Q      = mpcParameters.Q
        self.Qf     = mpcParameters.Qf
        self.R      = mpcParameters.R
        self.dR     = mpcParameters.dR
        self.n      = mpcParameters.n
        self.d      = mpcParameters.d
        self.Fx     = mpcParameters.Fx
        self.Fu     = mpcParameters.Fu
        self.bx     = mpcParameters.bx
        self.bu     = mpcParameters.bu
        self.xRef   = mpcParameters.xRef
        self.m      = predictiveModel.m

        self.slacks          = mpcParameters.slacks
        self.slackweight     = None
        self.timeVarying     = mpcParameters.timeVarying
        self.predictiveModel = predictiveModel
        self.osqp = None
        self.BT = None
        self.totalx = 0
        self.totalu = 0
        self.ndx = {}
        self.ndu = {}

        self.xPred = None
        self.uPred = None
        self.xLin = None
        self.uLin = None
        self.OldInput = np.zeros(self.d)

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.timeStep = 0
    def inittree(self,x,z):
        '''
        Initialize the scenario tree, each node contains the z prediction, x prediction (from last iteration) and u prediction (from last iteration), and weight
        '''
        u = np.zeros(self.d)
        self.BT = BranchTree(np.reshape(x,[1,self.n]),np.reshape(z,[1,self.n]),np.reshape(u,[1,self.d]),1,0)
        q = [self.BT]
        countx = 0
        countu = 0
        self.uLin = np.reshape(u,[1,self.d])
        self.xLin = np.reshape(x,[1,self.n])

        self.ndx[self.BT] = countx
        self.ndu[self.BT] = countu
        A,B,C,xp = self.predictiveModel.dyn_linearization(x,u)
        self.BT.dynmatr[0] = (A,B,C)
        countx+=self.BT.xtraj.shape[0]
        countu+=self.BT.xtraj.shape[0]

        while len(q)>0:
            currentbranch = q.pop(0)

            if currentbranch.depth<self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1],currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp= dp
                for i in range(0,self.m):
                    xtraj = np.zeros((self.N,self.n))
                    utraj = np.zeros((self.N,self.d))
                    newbranch = BranchTree(xtraj,zPred[:,self.n*i:self.n*(i+1)],utraj,p[i]*currentbranch.w,currentbranch.depth+1)
                    A,B,C,xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],currentbranch.utraj[-1])
                    newbranch.xtraj[0] = xp
                    for t in range(0,self.N):
                        A,B,C,xp = self.predictiveModel.dyn_linearization(newbranch.xtraj[t],newbranch.utraj[t])
                        newbranch.dynmatr[t] = (A,B,C)
                        if t<self.N-1:
                            newbranch.xtraj[t+1] = xp

                    self.ndx[newbranch] = countx
                    self.ndu[newbranch] = countu

                    self.xLin = np.vstack((self.xLin,newbranch.xtraj))
                    self.uLin = np.vstack((self.uLin,newbranch.utraj))
                    if newbranch.depth == self.NB:
                        countx+=(newbranch.xtraj.shape[0]+1)
                    else:
                        countx+=newbranch.xtraj.shape[0]
                    countu+=newbranch.xtraj.shape[0]
                    currentbranch.addchild(newbranch)
                    q.append(newbranch)
        self.totalx = countx
        self.totalu = countu
        self.slackweight = np.zeros(self.totalx*(self.Fx.shape[0]+1))


    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.totalx*self.n)
        Gu = np.zeros((self.totalx*self.n, self.totalu*self.d))

        E = np.zeros((self.totalx*self.n, self.n))
        E[0:self.n] = np.eye(self.n)

        L = np.zeros(self.totalx*self.n)
        self.E = E

        totalxdim = self.totalx*self.n
        for branch in self.ndx:
            l = branch.xtraj.shape[0]
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            for t in range(1,l):
                A,B,C = branch.dynmatr[t-1]
                Gx[(ndx+t)*self.n:(ndx+t+1)*self.n,(ndx+t-1)*self.n:(ndx+t)*self.n] = -A
                Gu[(ndx+t)*self.n:(ndx+t+1)*self.n,(ndu+t-1)*self.d:(ndu+t)*self.d] = -B
                L[(ndx+t)*self.n:(ndx+t+1)*self.n]                                  = C
            A,B,C = branch.dynmatr[-1]
            if branch.depth<self.NB:
                for child in branch.children:
                    ndxc = self.ndx[child]
                    Gx[ndxc*self.n:(ndxc+1)*self.n,(ndx+l-1)*self.n:(ndx+l)*self.n] = -A
                    Gu[ndxc*self.n:(ndxc+1)*self.n,(ndu+l-1)*self.d:(ndu+l)*self.d] = -B
                    L[ndxc*self.n:(ndxc+1)*self.n]                                  = C
            else:
                Gx[(ndx+l)*self.n:(ndx+l+1)*self.n,(ndx+l-1)*self.n:(ndx+l)*self.n] = -A
                Gu[(ndx+l)*self.n:(ndx+l+1)*self.n,(ndu+l-1)*self.d:(ndu+l)*self.d] = -B
                L[(ndx+l)*self.n:(ndx+l+1)*self.n]                                  = C
        self.L = L

        if self.slacks == True:
            self.G = np.hstack( (Gx, Gu, np.zeros( ( Gx.shape[0], self.slackweight.shape[0]) ) ) )
        else:
            self.G = np.hstack((Gx, Gu))

    def updatetree(self,x,z):
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            branch.utraj[0:l-1] = self.uLin[self.ndu[branch]+1:self.ndu[branch]+l]
            if branch.depth<self.NB:
                idx = np.argmax(branch.p)
                ndu = self.ndu[branch.children[idx]]
                branch.utraj[-1] = self.uLin[ndu]
            else:
                branch.utraj[-1] = branch.utraj[-2]
        self.BT.ztraj = np.reshape(z,[1,self.n])
        self.BT.xtraj = np.reshape(x,[1,self.n])
        q = [self.BT]

        while len(q)>0:
            currentbranch = q.pop(0)
            if currentbranch.depth<self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1],currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp = dp
                for i in range(0,self.m):
                    child = currentbranch.children[i]
                    child.w = currentbranch.w*p[i]
                    child.ztraj = zPred[:,i*self.n:(i+1)*self.n]
                    xtraj = np.zeros((self.N,self.n))
                    A,B,C,xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],currentbranch.utraj[-1])
                    child.xtraj[0] = xp
                    for t in range(0,self.N):
                        A,B,C,xp = self.predictiveModel.dyn_linearization(child.xtraj[t],child.utraj[t])
                        child.dynmatr[t] = (A,B,C)
                        if t<self.N-1:
                            child.xtraj[t+1] = xp

                    q.append(child)


    def buildCost(self):
        totalxdim = self.totalx*self.n
        listQ = [None] * (self.totalx)
        Hu = np.zeros([self.totalu*self.d,self.totalu*self.d])
        dRmat = np.diag(self.dR)
        qx = np.zeros(self.totalx*self.n)
        for branch in self.ndx:
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            l = branch.utraj.shape[0]
            for i in range(0,l-1):
                t = 1+self.N*(branch.depth-1)+i
                listQ[ndx+i]=self.Q*branch.w
                qx[(ndx+i)*self.n:(ndx+i+1)*self.n] = -2*branch.w*np.dot(self.xRef,self.Q)
                Hu[(ndu+i)*self.d:(ndu+i+1)*self.d,(ndu+i)*self.d:(ndu+i+1)*self.d] = branch.w*self.R

            if branch.depth<self.NB:
                Hu[(ndu+l-1)*self.d:(ndu+l)*self.d,(ndu+l-1)*self.d:(ndu+l)*self.d] = branch.w*self.R

                listQ[ndx+l-1] = self.Q*branch.w
                childJ = np.zeros(self.m)
                for j in range(0,self.m):
                    childJ[j] = branch.children[j].J

                qx[(ndx+l-1)*self.n:(ndx+l)*self.n] = branch.w*(-2*np.dot(self.xRef,self.Q)+np.dot(childJ,branch.dp))

            else:
                Hu[(ndu+l-1)*self.d:(ndu+l)*self.d,(ndu+l-1)*self.d:(ndu+l)*self.d] = branch.w*self.R
                listQ[ndx+l-1] = self.Q*branch.w
                listQ[ndx+l] = self.Qf*branch.w
                qx[(ndx+l-1)*self.n:(ndx+l)*self.n] = -2*branch.w*np.dot(self.xRef,self.Qf)

        Hx = linalg.block_diag(*listQ)
        qu = np.zeros(self.totalu*self.d)
        qu[0:self.d] = -2*self.OldInput @ self.dR

        # Cost linear term for state and input
        q = np.append(qx,qu)

        if self.slacks == True:
            quadSlack = self.Qslack[0] * np.eye(self.slackweight.shape[0])
            linSlack  = self.Qslack[1] * self.slackweight
            self.H = linalg.block_diag(Hx, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else:
            self.H = linalg.block_diag(Hx, Hu)
            self.q = q
        self.H = 2*self.H  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b
        # Let's start by computing the submatrix of F relates with the state
        Nc = self.Fx.shape[0]+1
        slackweight_x = np.zeros(self.totalx*Nc)


        Fxtot = np.zeros([Nc*self.totalx,self.totalx*self.n])
        bxtot = np.zeros(Nc*self.totalx)
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0,l):
                # collision constraint linearized
                h,dh = self.predictiveModel.col_eval(branch.xtraj[i],branch.ztraj[i])
                idx = self.ndx[branch]+i
                Fxtot[idx*Nc:(idx+1)*Nc,idx*self.n:(idx+1)*self.n] = np.vstack((-dh,self.Fx))
                bxtot[idx*Nc:(idx+1)*Nc] = np.append(h,self.bx)
                slackweight_x[idx*Nc:(idx+1)*Nc] = branch.w

        self.slackweight = slackweight_x
        rep_b = [self.Fu] * (self.totalu)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.totalu)

        # Let's stack all together
        F_hard = linalg.block_diag(Fxtot, Futot)

        # Add slack if need

        if self.slacks == True:
            nc_x = Fxtot.shape[0] # add slack only for state constraints
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc_x))
            addSlack[0:nc_x, 0:nc_x] = -np.eye(nc_x)
            # Now constraint slacks >= 0
            I = - np.eye(nc_x); Zeros = np.zeros((nc_x, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = np.vstack(( np.hstack((F_hard, addSlack)) , Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x)))
        else:
            self.F = F_hard
            self.b = np.hstack((bxtot, butot))
    def updateIneqConstr(self):
        # for warm_start, faster than building the constraints from scratch
        Nc = self.Fx.shape[0]+1
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0,l):
                h,dh = self.predictiveModel.col_eval(branch.xtraj[i],branch.ztraj[i])
                idx = self.ndx[branch]+i
                self.F[idx*Nc,idx*self.n:(idx+1)*self.n] = -dh
                self.b[idx*Nc] = h
                self.slackweight[idx*Nc:(idx+1)*Nc] = branch.w


    def solve(self, x,z,xRef=None):
        """Computes control action
        Arguments:
            x: ego vehicle state
            z: uncontrolled vehicle state
        """

        if not xRef is None:
            self.xRef = xRef
        if self.BT is None:
            self.inittree(x,z)
            self.buildIneqConstr()
        else:
            self.updatetree(x,z)
            self.updateIneqConstr()

        self.buildCost()
        self.buildEqConstr()


        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L
        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,x),self.L_FTOCP))
        self.unpackSolution()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        # print("Solver Time: ", self.solverTime.total_seconds(), " seconds.")


        # update applied input
        self.OldInput = self.uPred[0,:]
        self.timeStep += 1


    def addTerminalComponents(self):
        # TO DO: ....
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L

    def unpackSolution(self):
        # Extract predicted state and predicted input trajectories
        if self.feasible:
            self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.totalx*self.n)]),(-1,self.n)))).T
            self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.totalx*self.n+np.arange(self.totalu*self.d)]),(-1, self.d)))).T
            self.xLin = self.xPred
            self.uLin = self.uPred
            self.uLin = np.vstack((self.uLin,self.uLin[-1]))

    def BT2array(self):
        # for plotting and dubug maining, outputs the scenario tree as x and z trajectories
        ztraj = []
        xtraj = []
        utraj = []
        q = [self.BT]
        branch_w = []
        while (len(q)>0):
            curr = q.pop(0)
            for child in curr.children:
                branch_w.append(child.w)
                ztraj.append(np.vstack((curr.ztraj[-1],child.ztraj)))
                xtraj.append(np.vstack((curr.xtraj[-1],child.xtraj)))
                utraj.append(np.vstack((curr.utraj[-1],child.utraj)))
                q.append(child)
        return xtraj,ztraj,utraj,branch_w

    def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """
        qp_A = vstack([G, A]).tocsc()
        l = -inf * ones(len(h))
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])

        self.osqp = OSQP()

        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0

        self.Solution = res.x

class BranchMPC():

    def __init__(self,  mpcParameters, predictiveModel):
        """Initialization
        Arguments:
            mpcParameters: struct containing MPC parameters
            predictiveModel: containing CasADi functions about z prediction, collision constraints, and branching probability
        """
        self.N      = mpcParameters.N
        self.NB     = mpcParameters.NB
        self.Qslack = mpcParameters.Qslack
        self.Q      = mpcParameters.Q
        self.Qf     = mpcParameters.Qf
        self.R      = mpcParameters.R
        self.dR     = mpcParameters.dR
        self.n      = mpcParameters.n
        self.d      = mpcParameters.d
        self.Fx     = mpcParameters.Fx
        self.Fu     = mpcParameters.Fu
        self.bx     = mpcParameters.bx
        self.bu     = mpcParameters.bu
        self.xRef   = mpcParameters.xRef
        self.m      = predictiveModel.m

        self.slacks          = mpcParameters.slacks
        self.slackweight     = None
        self.timeVarying     = mpcParameters.timeVarying
        self.predictiveModel = predictiveModel
        self.osqp = None
        self.BT = None
        self.totalx = 0
        self.totalu = 0
        self.ndx = {}
        self.ndu = {}

        self.xPred = None
        self.uPred = None
        self.xLin = None
        self.uLin = None
        self.OldInput = np.zeros(self.d)

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.timeStep = 0
    def inittree(self,x,z):
        '''
        Initialize the scenario tree, each node contains the z prediction, x prediction (from last iteration) and u prediction (from last iteration), and weight
        '''
        u = np.zeros(self.d)
        self.BT = BranchTree(np.reshape(x,[1,self.n]),np.reshape(z,[1,self.n]),np.reshape(u,[1,self.d]),1,0)
        q = [self.BT]
        countx = 0
        countu = 0
        self.uLin = np.reshape(u,[1,self.d])
        self.xLin = np.reshape(x,[1,self.n])

        self.ndx[self.BT] = countx
        self.ndu[self.BT] = countu
        A,B,C,xp = self.predictiveModel.dyn_linearization(x,u)
        self.BT.dynmatr[0] = (A,B,C)
        countx+=self.BT.xtraj.shape[0]
        countu+=self.BT.xtraj.shape[0]

        while len(q)>0:
            currentbranch = q.pop(0)

            if currentbranch.depth<self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1],currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp= dp
                for i in range(0,self.m):
                    xtraj = np.zeros((self.N,self.n))
                    utraj = np.zeros((self.N,self.d))
                    newbranch = BranchTree(xtraj,zPred[:,self.n*i:self.n*(i+1)],utraj,p[i]*currentbranch.w,currentbranch.depth+1)
                    A,B,C,xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],currentbranch.utraj[-1])
                    newbranch.xtraj[0] = xp
                    for t in range(0,self.N):
                        A,B,C,xp = self.predictiveModel.dyn_linearization(newbranch.xtraj[t],newbranch.utraj[t])
                        newbranch.dynmatr[t] = (A,B,C)
                        if t<self.N-1:
                            newbranch.xtraj[t+1] = xp

                    self.ndx[newbranch] = countx
                    self.ndu[newbranch] = countu

                    self.xLin = np.vstack((self.xLin,newbranch.xtraj))
                    self.uLin = np.vstack((self.uLin,newbranch.utraj))
                    if newbranch.depth == self.NB:
                        countx+=(newbranch.xtraj.shape[0]+1)
                    else:
                        countx+=newbranch.xtraj.shape[0]
                    countu+=newbranch.xtraj.shape[0]
                    currentbranch.addchild(newbranch)
                    q.append(newbranch)
        self.totalx = countx
        self.totalu = countu
        self.slackweight = np.zeros(self.totalx*(self.Fx.shape[0]+1))


    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.totalx*self.n)
        Gu = np.zeros((self.totalx*self.n, self.totalu*self.d))

        E = np.zeros((self.totalx*self.n, self.n))
        E[0:self.n] = np.eye(self.n)

        L = np.zeros(self.totalx*self.n)
        self.E = E

        totalxdim = self.totalx*self.n
        for branch in self.ndx:
            l = branch.xtraj.shape[0]
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            for t in range(1,l):
                A,B,C = branch.dynmatr[t-1]
                Gx[(ndx+t)*self.n:(ndx+t+1)*self.n,(ndx+t-1)*self.n:(ndx+t)*self.n] = -A
                Gu[(ndx+t)*self.n:(ndx+t+1)*self.n,(ndu+t-1)*self.d:(ndu+t)*self.d] = -B
                L[(ndx+t)*self.n:(ndx+t+1)*self.n]                                  = C
            A,B,C = branch.dynmatr[-1]
            if branch.depth<self.NB:
                for child in branch.children:
                    ndxc = self.ndx[child]
                    Gx[ndxc*self.n:(ndxc+1)*self.n,(ndx+l-1)*self.n:(ndx+l)*self.n] = -A
                    Gu[ndxc*self.n:(ndxc+1)*self.n,(ndu+l-1)*self.d:(ndu+l)*self.d] = -B
                    L[ndxc*self.n:(ndxc+1)*self.n]                                  = C
            else:
                Gx[(ndx+l)*self.n:(ndx+l+1)*self.n,(ndx+l-1)*self.n:(ndx+l)*self.n] = -A
                Gu[(ndx+l)*self.n:(ndx+l+1)*self.n,(ndu+l-1)*self.d:(ndu+l)*self.d] = -B
                L[(ndx+l)*self.n:(ndx+l+1)*self.n]                                  = C
        self.L = L

        if self.slacks == True:
            self.G = np.hstack( (Gx, Gu, np.zeros( ( Gx.shape[0], self.slackweight.shape[0]) ) ) )
        else:
            self.G = np.hstack((Gx, Gu))

    def updatetree(self,x,z):
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            branch.utraj[0:l-1] = self.uLin[self.ndu[branch]+1:self.ndu[branch]+l]
            if branch.depth<self.NB:
                idx = np.argmax(branch.p)
                ndu = self.ndu[branch.children[idx]]
                branch.utraj[-1] = self.uLin[ndu]
            else:
                branch.utraj[-1] = branch.utraj[-2]
        self.BT.ztraj = np.reshape(z,[1,self.n])
        self.BT.xtraj = np.reshape(x,[1,self.n])
        for i in range(0,self.BT.xtraj.shape[0]):
            A,B,C,xp = self.predictiveModel.dyn_linearization(self.BT.xtraj[i],self.BT.utraj[i])
            self.BT.dynmatr[i]=(A,B,C)
        q = [self.BT]

        while len(q)>0:
            currentbranch = q.pop(0)
            if currentbranch.depth<self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1],currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp = dp
                for i in range(0,self.m):
                    child = currentbranch.children[i]
                    child.w = currentbranch.w*p[i]
                    child.ztraj = zPred[:,i*self.n:(i+1)*self.n]
                    xtraj = np.zeros((self.N,self.n))
                    A,B,C,xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],currentbranch.utraj[-1])
                    child.xtraj[0] = xp
                    for t in range(0,self.N):
                        A,B,C,xp = self.predictiveModel.dyn_linearization(child.xtraj[t],child.utraj[t])
                        child.dynmatr[t] = (A,B,C)
                        if t<self.N-1:
                            child.xtraj[t+1] = xp

                    q.append(child)


    def buildCost(self):
        totalxdim = self.totalx*self.n
        listQ = [None] * (self.totalx)
        Hu = np.zeros([self.totalu*self.d,self.totalu*self.d])
        dRmat = np.diag(self.dR)
        qx = np.zeros(self.totalx*self.n)
        dQ = self.Q*0.5
        for branch in self.ndx:
            ndx = self.ndx[branch]
            ndu = self.ndu[branch]
            l = branch.utraj.shape[0]
            for i in range(0,l-1):
                t = 1+self.N*(branch.depth-1)+i
                listQ[ndx+i]=(dQ+self.Q)*branch.w
                qx[(ndx+i)*self.n:(ndx+i+1)*self.n] = -2*branch.w*(np.dot(self.xRef,self.Q)+np.dot(branch.xtraj[i],dQ))
                Hu[(ndu+i)*self.d:(ndu+i+1)*self.d,(ndu+i)*self.d:(ndu+i+1)*self.d] = branch.w*self.R

            if branch.depth<self.NB:
                Hu[(ndu+l-1)*self.d:(ndu+l)*self.d,(ndu+l-1)*self.d:(ndu+l)*self.d] = branch.w*self.R

                listQ[ndx+l-1] = (dQ+self.Q)*branch.w
                childJ = np.zeros(self.m)
                for j in range(0,self.m):
                    childJ[j] = branch.children[j].J

                qx[(ndx+l-1)*self.n:(ndx+l)*self.n] = branch.w*(-2*np.dot(self.xRef,self.Q)-2*np.dot(branch.xtraj[-1],dQ)+np.dot(childJ,branch.dp))

            else:
                Hu[(ndu+l-1)*self.d:(ndu+l)*self.d,(ndu+l-1)*self.d:(ndu+l)*self.d] = branch.w*self.R
                listQ[ndx+l-1] = (dQ+self.Q)*branch.w
                listQ[ndx+l] = self.Qf*branch.w
                qx[(ndx+l-1)*self.n:(ndx+l)*self.n] = -2*branch.w*(np.dot(self.xRef,self.Qf)+np.dot(branch.xtraj[-1],dQ))

        Hx = linalg.block_diag(*listQ)
        qu = np.zeros(self.totalu*self.d)
        qu[0:self.d] = -2*self.OldInput @ self.dR

        # Cost linear term for state and input
        q = np.append(qx,qu)

        if self.slacks == True:
            quadSlack = self.Qslack[0] * np.eye(self.slackweight.shape[0])
            linSlack  = self.Qslack[1] * self.slackweight
            self.H = linalg.block_diag(Hx, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else:
            self.H = linalg.block_diag(Hx, Hu)
            self.q = q
        self.H = 2*self.H  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b
        # Let's start by computing the submatrix of F relates with the state
        Nc = self.Fx.shape[0]+1
        slackweight_x = np.zeros(self.totalx*Nc)


        Fxtot = np.zeros([Nc*self.totalx,self.totalx*self.n])
        bxtot = np.zeros(Nc*self.totalx)
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0,l):
                # collision constraint linearized
                h,dh = self.predictiveModel.col_eval(branch.xtraj[i],branch.ztraj[i])
                idx = self.ndx[branch]+i
                Fxtot[idx*Nc:(idx+1)*Nc,idx*self.n:(idx+1)*self.n] = np.vstack((-dh,self.Fx))
                bxtot[idx*Nc:(idx+1)*Nc] = np.append(h,self.bx)
                slackweight_x[idx*Nc:(idx+1)*Nc] = branch.w

        self.slackweight = slackweight_x
        rep_b = [self.Fu] * (self.totalu)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.totalu)

        # Let's stack all together
        F_hard = linalg.block_diag(Fxtot, Futot)

        # Add slack if need

        if self.slacks == True:
            nc_x = Fxtot.shape[0] # add slack only for state constraints
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc_x))
            addSlack[0:nc_x, 0:nc_x] = -np.eye(nc_x)
            # Now constraint slacks >= 0
            I = - np.eye(nc_x); Zeros = np.zeros((nc_x, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = np.vstack(( np.hstack((F_hard, addSlack)) , Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x)))
        else:
            self.F = F_hard
            self.b = np.hstack((bxtot, butot))
    def updateIneqConstr(self):
        # for warm_start, faster than building the constraints from scratch
        Nc = self.Fx.shape[0]+1
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            for i in range(0,l):
                h,dh = self.predictiveModel.col_eval(branch.xtraj[i],branch.ztraj[i])
                idx = self.ndx[branch]+i
                self.F[idx*Nc,idx*self.n:(idx+1)*self.n] = -dh
                self.b[idx*Nc] = h
                self.slackweight[idx*Nc:(idx+1)*Nc] = branch.w


    def solve(self, x,z,xRef=None):
        """Computes control action
        Arguments:
            x: ego vehicle state
            z: uncontrolled vehicle state
        """

        if not xRef is None:
            self.xRef = xRef
        if self.BT is None:
            self.inittree(x,z)
            self.buildIneqConstr()
        else:
            self.updatetree(x,z)
            self.updateIneqConstr()

        self.buildCost()
        self.buildEqConstr()


        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L
        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,x),self.L_FTOCP))
        self.unpackSolution()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        # print("Solver Time: ", self.solverTime.total_seconds(), " seconds.")


        # update applied input
        self.OldInput = self.uPred[0,:]
        self.timeStep += 1


    def addTerminalComponents(self):
        # TO DO: ....
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L

    def unpackSolution(self):
        # Extract predicted state and predicted input trajectories
        if self.feasible:
            self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.totalx*self.n)]),(-1,self.n)))).T
            self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.totalx*self.n+np.arange(self.totalu*self.d)]),(-1, self.d)))).T
            self.xLin = self.xPred
            self.uLin = self.uPred
            self.uLin = np.vstack((self.uLin,self.uLin[-1]))

    def BT2array(self):
        # for plotting and dubug maining, outputs the scenario tree as x and z trajectories
        ztraj = []
        xtraj = []
        utraj = []
        branch_w = []
        q = [self.BT]
        while (len(q)>0):
            curr = q.pop(0)
            for child in curr.children:
                branch_w.append(child.w)
                ztraj.append(np.vstack((curr.ztraj[-1],child.ztraj)))
                xtraj.append(np.vstack((curr.xtraj[-1],child.xtraj)))
                utraj.append(np.vstack((curr.utraj[-1],child.utraj)))
                q.append(child)
        return xtraj,ztraj,utraj,branch_w

    def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """
        qp_A = vstack([G, A]).tocsc()
        l = -inf * ones(len(h))
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])

        self.osqp = OSQP()

        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)
        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0

        self.Solution = res.x
class robustMPC():
    """ robust Model Predicitve Controller class
        for benchmark, avoiding all branches in the scenario tree
    """
    def __init__(self,  mpcParameters, predictiveModel):
        """Initialization
        Arguments:
            mpcParameters: struct containing MPC parameters
        """
        self.N      = mpcParameters.N
        self.NB     = mpcParameters.NB
        self.Qslack = mpcParameters.Qslack
        self.Q      = mpcParameters.Q
        self.Qf     = mpcParameters.Qf
        self.R      = mpcParameters.R
        self.dR     = mpcParameters.dR
        self.n      = mpcParameters.n
        self.d      = mpcParameters.d
        self.A      = mpcParameters.A
        self.B      = mpcParameters.B
        self.Fx     = mpcParameters.Fx
        self.Fu     = mpcParameters.Fu
        self.bx     = mpcParameters.bx
        self.bu     = mpcParameters.bu
        self.xRef   = mpcParameters.xRef
        self.m      = predictiveModel.m
        self.Nx     = self.N*self.NB+2
        self.Nu     = self.N*self.NB+1

        self.slacks          = mpcParameters.slacks
        self.slackdim        = None
        self.timeVarying     = mpcParameters.timeVarying
        self.predictiveModel = predictiveModel
        self.osqp = None
        self.BT = None
        self.zPred = [None]*(self.N*self.NB+1)
        self.zcount = 0

        self.OldInput = np.zeros((1,2)) # TO DO fix size


        self.xPred = None
        self.uLin = None

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.timeStep = 0

    def get_xLin(self,x):
        if self.uLin is None:
            self.uLin = np.zeros([self.Nu,self.d])
        self.uLin = np.vstack((self.uLin,self.uLin[-1]))
        self.xLin = np.zeros([self.Nx,self.n])

        self.xLin[0] = x
        for i in range(0,self.Nx-1):
            A,B,C,xp=self.predictiveModel.dyn_linearization(self.xLin[i],self.uLin[i])
            self.xLin[i+1] = xp
    def inittree(self,z):
        self.BT = BranchTree(np.empty((1,self.n)),np.reshape(z,[1,self.n]),np.empty((1,self.d)),1,0)
        q = [self.BT]
        for t in range(0,len(self.zPred)):
            self.zPred[t] = np.empty([0,self.n])
        self.zPred[0] = np.array([z])
        self.zcount=1
        while len(q)>0:
            currentbranch = q.pop(0)
            if currentbranch.depth>0:
                for i in range(0,currentbranch.ztraj.shape[0]):
                    t = (currentbranch.depth-1)*self.N+i+1
                    self.zPred[t] = np.vstack((self.zPred[t],currentbranch.ztraj[i]))
                    self.zcount+=1

            if currentbranch.depth<self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1],currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp= dp

                for i in range(0,self.m):
                    newbranch = BranchTree(np.empty((self.N,self.n)),zPred[:,self.n*i:self.n*(i+1)],np.empty((self.N,self.d)),p[i]*currentbranch.w,currentbranch.depth+1)
                    currentbranch.addchild(newbranch)
                    q.append(newbranch)

    def updatetree(self,z):
        q = [self.BT]
        self.BT.ztraj = np.reshape(z,[1,self.n])
        for t in range(0,len(self.zPred)):
            self.zPred[t] = np.empty([0,self.n])
        self.zPred[0] = np.array([z])
        while len(q)>0:
            currentbranch = q.pop(0)
            if currentbranch.depth>0:
                for i in range(0,currentbranch.ztraj.shape[0]):
                    t = (currentbranch.depth-1)*self.N+i+1
                    self.zPred[t] = np.vstack((self.zPred[t],currentbranch.ztraj[i]))
            if currentbranch.depth<self.NB:
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1],currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp= dp

                for i in range(0,self.m):
                    currentbranch.children[i].ztraj = zPred[:,self.n*i:self.n*(i+1)]
                    currentbranch.children[i].w = p[i]*currentbranch.w
                    q.append(currentbranch.children[i])

    def BT2array(self):
        ztraj = []
        # xtraj = []
        # utraj = []
        q = [self.BT]
        while (len(q)>0):
            curr = q.pop(0)
            for child in curr.children:
                ztraj.append(np.vstack((curr.ztraj[-1],child.ztraj)))
                q.append(child)

        return [self.xPred],ztraj,[self.uPred],[]
    def solve(self, x,z,xRef=None):
        """Computes control action
        Arguments:
            x0: current state
        """

        if not xRef is None:
            self.xRef = xRef
        if self.BT is None:
            self.inittree(z)
            self.get_xLin(x)
        else:
            self.updatetree(z)
        self.computeLTVdynamics()
        self.buildIneqConstr()
        self.buildCost()
        self.buildEqConstr()
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L
        # Solve QP
        startTimer = datetime.datetime.now()
        self.osqp_solve_qp(self.H_FTOCP, self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,x),self.L_FTOCP))
        self.unpackSolution()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.feasibleStateInput()
        # If LTV active --> compute state-input linearization trajectory
        if self.timeVarying == True:
            self.xLin = np.vstack((self.xPred[1:, :], self.zt))
            self.uLin = np.vstack((self.uPred[1:, :], self.zt_u))

        # update applied input
        self.OldInput = self.uPred[0,:]
        self.timeStep += 1


    def computeLTVdynamics(self):
        # Estimate system dynamics
        self.A = []; self.B = []; self.C =[]; self.h0 = []; self.Jh = []
        for i in range(0, self.Nu):
            Ai, Bi, Ci, xpi = self.predictiveModel.dyn_linearization(self.xLin[i], self.uLin[i])
            self.A.append(Ai); self.B.append(Bi); self.C.append(Ci);

    def addTerminalComponents(self, x0):
        # TO DO: ....
        self.H_FTOCP = sparse.csc_matrix(self.H)
        self.q_FTOCP = self.q
        self.F_FTOCP = sparse.csc_matrix(self.F)
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)
        self.E_FTOCP = self.E
        self.L_FTOCP = self.L

    def feasibleStateInput(self):
        self.zt   = self.xPred[-1,:]
        self.zt_u = self.uPred[-1,:]

    def unpackSolution(self):
        # Extract predicted state and predicted input trajectories
        self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.n*self.Nx)]),(self.Nx,self.n)))).T
        self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.n*self.Nx+np.arange(self.d*self.Nu)]),(self.Nu, self.d)))).T
        self.xLin = self.xPred
        self.uLin = self.uPred
        self.uLin = np.vstack((self.uLin,self.uLin[-1]))

    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b
        # Let's start by computing the submatrix of F relates with the state
        rep_a = [self.Fx] * (self.Nx)
        Fxtot = linalg.block_diag(*rep_a)
        bxtot = np.tile(np.squeeze(self.bx), self.Nx)

        Fxbackup = np.zeros([self.zcount,self.Nx*self.n])
        bxbackup = np.zeros(self.zcount)
        counter = 0
        for i in range(0,len(self.zPred)):
            for j in range(0,self.zPred[i].shape[0]):
                h,dh = self.predictiveModel.col_eval(self.xLin[i],self.zPred[i][j])
                Fxbackup[counter][self.n*i:self.n*(i+1)] = -dh
                bxbackup[counter] = h
                counter+=1

        Fxtot = np.vstack((Fxtot,Fxbackup[0:counter]))
        bxtot = np.append(bxtot,bxbackup[0:counter])
        self.slackdim = Fxtot.shape[0]
        # Let's start by computing the submatrix of F relates with the input
        rep_b = [self.Fu] * (self.Nu)
        Futot = linalg.block_diag(*rep_b)
        butot = np.tile(np.squeeze(self.bu), self.Nu)

        # Let's stack all together
        F_hard = linalg.block_diag(Fxtot, Futot)

        # Add slack if need
        if self.slacks == True:
            nc_x = Fxtot.shape[0] # add slack only for state constraints
            # Fist add add slack to existing constraints
            addSlack = np.zeros((F_hard.shape[0], nc_x))
            addSlack[0:nc_x, 0:nc_x] = -np.eye(nc_x)
            # Now constraint slacks >= 0
            I = - np.eye(nc_x); Zeros = np.zeros((nc_x, F_hard.shape[1]))
            Positivity = np.hstack((Zeros, I))

            # Let's stack all together
            self.F = np.vstack(( np.hstack((F_hard, addSlack)) , Positivity))
            self.b = np.hstack((bxtot, butot, np.zeros(nc_x)))
        else:
            self.F = F_hard
            self.b = np.hstack((bxtot, butot))

    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.n * self.Nx)
        Gu = np.zeros((self.n * self.Nx, self.d * self.Nu))

        E = np.zeros((self.n * self.Nx, self.n))
        E[np.arange(self.n)] = np.eye(self.n)

        L = np.zeros(self.n * self.Nx)

        for i in range(0, self.Nu):
            if self.timeVarying == True:
                Gx[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.n):(i*self.n + self.n)] = -self.A[i]
                Gu[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.d):(i*self.d + self.d)] = -self.B[i]
                L[(self.n + i*self.n):(self.n + i*self.n + self.n)]                                  =  self.C[i]
            else:
                Gx[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.n):(i*self.n + self.n)] = -self.A
                Gu[(self.n + i*self.n):(self.n + i*self.n + self.n), (i*self.d):(i*self.d + self.d)] = -self.B

        if self.slacks == True:
            self.G = np.hstack( (Gx, Gu, np.zeros( ( Gx.shape[0], self.slackdim) ) ) )
        else:
            self.G = np.hstack((Gx, Gu))

        self.E = E
        self.L = L

    def buildCost(self):
        # The cost is: (1/2) * z' H z + q' z
        listQ = [self.Q] * (self.Nx-1)
        Hx = linalg.block_diag(*listQ)

        listTotR = [self.R + 2 * np.diag(self.dR)] * (self.Nu) # Need to add dR for the derivative input cost
        Hu = linalg.block_diag(*listTotR)
        # Need to condider that the last input appears just once in the difference
        for i in range(0, self.d):
            Hu[ i - self.d, i - self.d] = Hu[ i - self.d, i - self.d] - self.dR[i]

        # Derivative Input Cost
        OffDiaf = -np.tile(self.dR, self.Nu-1)
        np.fill_diagonal(Hu[self.d:], OffDiaf)
        np.fill_diagonal(Hu[:, self.d:], OffDiaf)

        # Cost linear term for state and input
        q = - 2 * np.dot(np.append(np.tile(self.xRef, self.Nx), np.zeros(self.R.shape[0] * self.Nu)), linalg.block_diag(Hx, self.Qf, Hu))
        # Derivative Input (need to consider input at previous time step)
        q[self.n*self.Nx:self.n*self.Nx+self.d] = -2 * np.dot( self.OldInput, np.diag(self.dR) )
        if self.slacks == True:
            quadSlack = self.Qslack[0] * np.eye(self.slackdim)
            linSlack  = self.Qslack[1] * np.ones(self.slackdim )
            self.H = linalg.block_diag(Hx, self.Qf, Hu, quadSlack)
            self.q = np.append(q, linSlack)
        else:
            self.H = linalg.block_diag(Hx, self.Qf, Hu)
            self.q = q

        self.H = 2 * self.H  #  Need to multiply by two because CVX considers 1/2 in front of quadratic cost

    def osqp_solve_qp(self, P, q, G= None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:
        minimize
            (1/2) * x.T * P * x + q.T * x
        subject to
            G * x <= h
            A * x == b
        using OSQP <https://github.com/oxfordcontrol/osqp>.
        """
        qp_A = vstack([G, A]).tocsc()
        l = -inf * ones(len(h))
        qp_l = hstack([l, b])
        qp_u = hstack([h, b])
        self.osqp = OSQP()
        self.osqp.setup(P=P, q=q, A=qp_A, l=qp_l, u=qp_u, verbose=False, polish=True)

        if initvals is not None:
            self.osqp.warm_start(x=initvals)
        res = self.osqp.solve()
        if res.info.status_val == 1:
            self.feasible = 1
        else:
            self.feasible = 0
        self.Solution = res.x

# 对应论文"IV. BRANCH MPC WITH RISK MEASURE OBJECTIVES"
class BranchMPC_CVaR():
    """Model Predicitve Controller class using CVaR objective
    """
    def __init__(self,  mpcParameters, predictiveModel,ralpha,S=None):
        """Initialization
        Arguments:(初始化参数)
            mpcParameters: struct containing MPC parameters
        """
        self.N      = mpcParameters.N       # number of time steps in between branches
        self.NB     = mpcParameters.NB      # number of branching(tree的最大深度,即branch的最大层数,根branch深度为0)
        self.Qslack = mpcParameters.Qslack  # it has to be a vector. Qslack = [linearSlackCost, quadraticSlackCost]
        self.Q      = mpcParameters.Q       # quadratic state cost
        self.Qf     = mpcParameters.Qf      # quadratic state cost final
        self.R      = mpcParameters.R       # quadratic input cost
        self.dR     = mpcParameters.dR      # Quadratic rate cost
        self.n      = mpcParameters.n       # dimension state space
        self.d      = mpcParameters.d       # dimension input space
        self.Fx     = mpcParameters.Fx      # State constraint Fx * x <= bx (每个状态节点都有约束,Fx是(4*4),bx是(4*1))
        self.Fu     = mpcParameters.Fu      # Input constraint Fu * u <= bu (Fu是(4*2),bu是(4*1))
        self.bx     = mpcParameters.bx      # 对应 Fx,状态约束的阈值向量 [max y,min y,max psi,min psi]
        self.bu     = mpcParameters.bu      # 对应 Fu,输入约束的阈值向量 [-min a,max a,-min r,max r]
        self.xRef   = mpcParameters.xRef    # sim_overtake()中,要lane change时的目标状态
        self.m      = predictiveModel.m     # number of policies(3个:maintain,brake,lc)
        self.psimax = mpcParameters.bx[0][2][0]
        self.S      = S              # for cross-product terms in the cost(应该是QP转化为SOCP用到的,最终会用在SOCP的不等式约束中)
        self.ralpha  = ralpha        # CVaR parameter,默认 0.1
        self.param  = mpcParameters

        # for the SOCP constraints, calculate the cholesky decomposition of Q and R matrices to encode the cost
        # 状态代价权重 Q cholesky 分解得到 self.Wx
        try:
            self.Wx = np.linalg.cholesky(self.Q).T  # 对 Q 进行 Cholesky 分解，得到下三角矩阵 Wx
        except np.linalg.linalg.LinAlgError:
            self.Wx = linalg.sqrtm(self.Q)          # 如果 Cholesky 分解失败，使用平方根分解
        # 输入代价权重 R cholesky 分解得到 self.Wu
        try:
            self.Wu = np.linalg.cholesky(self.R).T
        except np.linalg.linalg.LinAlgError:
            self.Wu = linalg.sqrtm(self.R)
        # 输入变化率代价权重 dR cholesky 分解得到 self.Wdu
        if not self.dR is None:
            dRmat = np.diag(self.dR)    # 将输入变化率成本 dR 转换为对角矩阵
            try:
                self.Wdu = np.linalg.cholesky(dRmat).T
            except np.linalg.linalg.LinAlgError:
                self.Wdu = linalg.sqrtm(dRmat)

        self.slacks          = mpcParameters.slacks
        self.slackweight     = None  # 整个tree上所有状态节点的slack weight,size:(totalx*(Fx.shape[0]+1), 1)
        self.timeVarying     = mpcParameters.timeVarying
        self.predictiveModel = predictiveModel
        self.osqp = None
        self.BT = None      # tree的根branch
        self.totalx = 0     # 记录整个tree中状态节点数
        self.totalu = 0     # 记录整个tree中输入节点数
        self.ndx = {}       # map:记录每个branch与其所包含的所有状态节点的起始索引
        self.ndu = {}       # map:记录每个branch与其所包含的所有输入节点的起始索引
        self.branchidx = {} # map:记录每个branch的索引(对branch编号,不同于状态节点的编号)
        self.branchdim = 0  # 整个tree中branch的数量

        self.xPred = None   # SOCP求解结果的(0 ~ self.totalx*self.n)部分,reshape为(-1,self.n),即self.totalx行,self.n列; 
        self.uPred = None   # SOCP求解结果的(self.totalx*self.n ~ self.totalx*self.n+self.totalu*self.d)不分,reshape为(-1,self.d),即self.totalu行,self.d列;
        self.xLin = None    # SOCP求解结果,也是self.xPred的转置为(n, totalx)的二维数组,每列表示一个状态。
        self.uLin = None    # SOCP求解结果,也是self.uPred的转置为(d, totalu)的二维数组,每列表示一个输入。
        self.OldInput = np.zeros(self.d)    # 记录总仿真时间T(10秒)内,每个仿真步长(0.1秒)的自车实际使用的输入

        # initialize time
        startTimer = datetime.datetime.now()
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        self.linearizationTime = deltaTimer
        self.timeStep = 0

    """
    功能: 初始化tree,包括场景树(scenario tree)和轨迹树(trajectory tree):其中scenario tree是最终结果;而trajectory tree只是生成了对应结构,并不是最终结果.
        1.通过广度优先搜索(BFS)构建树.以ego,obs当前状态x,z构建根branch,共拓展self.NB层(0,1,2共3层),每层共m个子branch。
        2.树的每个branch包含ztraj,w,depth,p,dp等信息,用于 Branch MPC 的优化计算.
        3.该tree为后续的 CVaR 计算和约束构建提供了基础。
    输入: x ego当前状态; z obs当前状态;
    """
    def inittree(self,x,z):
        # 初始化根branch(根branch就1个节点,对应ego与obs的状态就是他们的当前状态; 概率1; 深度0)
        u = np.zeros(self.d)
        self.BT = BranchTree(np.reshape(x,[1,self.n]), np.reshape(z,[1,self.n]), np.reshape(u,[1,self.d]), 1, 0)
        q = [self.BT]   # BFS广搜的队列
        countx = 0      # 记录整个tree中状态节点数(那么每个branch中包含N个状态节点的序号范围就是 countx ~ countx+N-1)
        countu = 0      # 同上
        countbranch = 0
        self.uLin = np.reshape(u,[1,self.d])
        self.xLin = np.reshape(x,[1,self.n])

        # 记录每个branch包含所有状态节点的起始索引(是一个map,key:branch对象,value:状态节点的起始索引)
        self.ndx[self.BT] = countx  # 记录根branch的起始节点索引
        self.ndu[self.BT] = countu  # 同上

        # 线性化动态模型
        A,B,C,xp = self.predictiveModel.dyn_linearization(x,u) # 返回A,B,C,xp(下一个状态)
        self.BT.dynmatr[0] = (A,B,C)
        countx+=self.BT.xtraj.shape[0]
        countu+=self.BT.xtraj.shape[0]

        # 使用广度优先搜索,生成scenario tree
        while len(q)>0:
            currentbranch = q.pop(0)

            # 如果当前branch深度小于branch最大数量(2)，继续扩展(根branch深度为0,所以有0,1,2共3层)
            if currentbranch.depth<self.NB:
                self.branchidx[currentbranch] = countbranch #用map给每个branch编号
                countbranch+=1
                # 1.生成obs的子branch:在obs状态z,m个策略下的子branch(包括每个branch的p,dp,w,depth,xtraj,utraj等信息)
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])    # size: (N, m*n)
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1], currentbranch.ztraj[-1])   # size: (m,1),因为对每个branch而言的
                currentbranch.p = p     # 每个子branch的概率p,累加=1(具体说明看论文)
                currentbranch.dp= dp
                # 2.初始化ego的子branch: 对utraj和xtraj只是先初始化,并不是最终的求解结果
                for i in range(0, self.m):
                    xtraj = np.zeros((self.N, self.n))
                    utraj = np.zeros((self.N, self.d))
                    newbranch = BranchTree(xtraj, zPred[:, self.n*i:self.n*(i+1)], utraj, p[i]*currentbranch.w, currentbranch.depth+1)
                    # 根据论文式(5b)求xp,即newbranch的初始状态与其父分支的最后一个状态之间，也要满足状态转移方程。
                    A,B,C,xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1], currentbranch.utraj[-1])
                    newbranch.xtraj[0] = xp
                    for t in range(0,self.N):
                        A,B,C,xp = self.predictiveModel.dyn_linearization(newbranch.xtraj[t],newbranch.utraj[t])
                        newbranch.dynmatr[t] = (A,B,C)
                        if t<self.N-1:
                            newbranch.xtraj[t+1] = xp

                    # 记录该newbranch包含的所有状态节点的起始索引
                    self.ndx[newbranch] = countx
                    self.ndu[newbranch] = countu

                    # 更新线性化状态和输入轨迹
                    self.xLin = np.vstack((self.xLin, newbranch.xtraj))
                    self.uLin = np.vstack((self.uLin, newbranch.utraj))
                    # 更新索引计数器
                    if newbranch.depth == self.NB:
                        countx+=(newbranch.xtraj.shape[0]+1)
                    else:
                        countx+=newbranch.xtraj.shape[0]
                    countu+=newbranch.xtraj.shape[0]
                    # 将newbranch加入currentbranch的子队列, 同时加入bfs的队列
                    currentbranch.addchild(newbranch)
                    q.append(newbranch)
        # 记录总状态和输入维度
        self.totalx = countx
        self.totalu = countu
        # 每个状态节点都有"self.Fx.shape[0]+1"个约束,故可得整个tree上所有状态节点的约束数量(等于slackweight的长度)
        self.slackweight = np.zeros(self.totalx*(self.Fx.shape[0]+1)) 
        self.branchdim = countbranch

    """
    构建等式约束.对应论文式(10),包括3部分:1.branch内的状态转移;2.branch与子branch之间的状态转移;3.风险度量的等式约束.
    """
    def buildEqConstr(self):
        # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: G*z = E * x(t) + L
        Gx = np.eye(self.totalx*self.n)
        Gu = np.zeros((self.totalx*self.n, self.totalu*self.d))

        E = np.zeros((self.totalx*self.n, self.n))
        E[0:self.n] = np.eye(self.n)

        L = np.zeros(self.totalx*self.n)


        totalxdim = self.totalx*self.n
        # 构建tree中所有点的状态转移方程的等式约束(包括branch内部,branch与子branch之间.tree的初始点b0不用,因此A,B,C都是从序号为(n,0)的子矩阵开始赋值。)
        for branch in self.ndx:
            l = branch.xtraj.shape[0]
            ndx = self.ndx[branch]      # branch上节点在整个tree中所有节点中的起始索引
            ndu = self.ndu[branch]
            # branch内的状态转移:从branch序号为1的点到最后一个点(序号从0开始),
            for t in range(1,l):
                A,B,C = branch.dynmatr[t-1]
                Gx[(ndx+t)*self.n:(ndx+t+1)*self.n, (ndx+t-1)*self.n:(ndx+t)*self.n] = -A
                Gu[(ndx+t)*self.n:(ndx+t+1)*self.n, (ndu+t-1)*self.d:(ndu+t)*self.d] = -B
                L[(ndx+t)*self.n:(ndx+t+1)*self.n]                                  = C
            # branch与子branch之间的状态转移
            A,B,C = branch.dynmatr[-1]
            if branch.depth<self.NB:
                for child in branch.children:
                    ndxc = self.ndx[child]
                    Gx[ndxc*self.n:(ndxc+1)*self.n, (ndx+l-1)*self.n:(ndx+l)*self.n] = -A
                    Gu[ndxc*self.n:(ndxc+1)*self.n, (ndu+l-1)*self.d:(ndu+l)*self.d] = -B
                    L[ndxc*self.n:(ndxc+1)*self.n]                                  = C
            else:
                Gx[(ndx+l)*self.n:(ndx+l+1)*self.n,(ndx+l-1)*self.n:(ndx+l)*self.n] = -A
                Gu[(ndx+l)*self.n:(ndx+l+1)*self.n,(ndu+l-1)*self.d:(ndu+l)*self.d] = -B
                L[(ndx+l)*self.n:(ndx+l+1)*self.n]                                  = C


        ## dual formulation of CVaR opt(对应论文式(10b),推导在附录中)
        # rho(i)=-s(i)+1/alpha*dot(mum(i,:),P(i,:)), (其中rho(i),s(i),alpha,mum(i,:)分别对应论文γi,σi,α,μi−)
        Arisk = np.zeros([self.branchdim, self.branchdim*(self.m*2+2)])
        for branch in self.branchidx:
            idx = self.branchidx[branch]
            Arisk[idx, idx] = 1
            Arisk[idx, self.branchdim+idx] = 1
            Arisk[idx, self.branchdim*(self.m+2)+idx*self.m:self.branchdim*(self.m+2)+(idx+1)*self.m]=-branch.p/self.ralpha
        self.G = linalg.block_diag(np.hstack((Gx,Gu)),Arisk)
        self.E = np.vstack((E,np.zeros((Arisk.shape[0],self.n))))
        self.L = np.append(L,np.zeros(Arisk.shape[0]))
        if self.slacks == True:
            self.G = np.hstack( ( self.G,np.zeros([self.G.shape[0],self.slackweight.shape[0]+1]) ) )
        else:
            self.G = np.hstack( ( self.G,np.zeros([self.G.shape[0],1]) ) )


    """
    输入: x ego当前状态; z obs当前状态;
    功能: 根据ego,obs当前时间步新的状态与上个时间步SOCP优化结果优化的结果self.uLin,对树中各个branch进行更新.
    """
    def updatetree(self,x,z):
        # 1.根据SOCP优化结果 self.uLin,更新每个branch的 utraj
        for branch in self.ndx:
            l = branch.utraj.shape[0]
            branch.utraj[0:l-1] = self.uLin[self.ndu[branch]+1:self.ndu[branch]+l]
            if branch.depth<self.NB:
                # 选择概率最大的子分支的第1个输入作为当前分支的utraj的最后一个输入
                idx = np.argmax(branch.p)
                ndu = self.ndu[branch.children[idx]]
                branch.utraj[-1] = self.uLin[ndu]
            else:
                # 对于深度等于 NB 的分支，使用当前utraj中倒数第二个输入作为最后一个输入
                branch.utraj[-1] = branch.utraj[-2]

        # 2.更新根branch:更新根branch中ztraj,xtraj(根branch状态序列的个数只有1个)
        self.BT.ztraj = np.reshape(z,[1,self.n])
        self.BT.xtraj = np.reshape(x,[1,self.n])
        # 更新根branch的动态模型参数dynmatr,即A,B,C
        for i in range(0,self.BT.xtraj.shape[0]):
            A,B,C,xp = self.predictiveModel.dyn_linearization(self.BT.xtraj[i],self.BT.utraj[i])
            self.BT.dynmatr[i]=(A,B,C)

        # 3.根据"ego,obs当前状态x,z,SOCP优化结果更新后的utraj",使用BFS更新整个tree(主要更新树各个分支除utraj外的其它量,包括p,dp,w,ztraj,xtraj,dynmatr等。)
        q = [self.BT]
        while len(q)>0:
            currentbranch = q.pop(0)
            if currentbranch.depth<self.NB:
                # 计算obs在m个策略下的预测轨迹和分支概率
                zPred = self.predictiveModel.zpred_eval(currentbranch.ztraj[-1])
                p,dp = self.predictiveModel.branch_eval(currentbranch.xtraj[-1],currentbranch.ztraj[-1])
                currentbranch.p = p
                currentbranch.dp = dp
                # 更新子分支
                for i in range(0,self.m):
                    child = currentbranch.children[i]
                    child.w = currentbranch.w*p[i]  # 更新子分支的权重
                    child.ztraj = zPred[:,i*self.n:(i+1)*self.n]    # 更新子分支中obs的预测轨迹
                    # 重新计算子分支的状态轨迹和动态模型参数
                    xtraj = np.zeros((self.N,self.n))
                    A,B,C,xp = self.predictiveModel.dyn_linearization(currentbranch.xtraj[-1],currentbranch.utraj[-1])
                    child.xtraj[0] = xp
                    for t in range(0,self.N):
                        A,B,C,xp = self.predictiveModel.dyn_linearization(child.xtraj[t],child.utraj[t])
                        child.dynmatr[t] = (A,B,C)
                        if t<self.N-1:
                            child.xtraj[t+1] = xp

                    q.append(child)


    """
    构建不等式约束矩阵 F 和对应的约束向量 b:
    1. 构建tree上所有节点状态x的Ineq约束: (1)Fxtot * x_tot <= bxtot(如果有S,就要考虑在Fxtot中); (2)self.slackweight
    2. 构建tree上所有节点输入u的Ineq约束: Futot * u_tot <= butot
    3. 构建tree上所有branch与 CVaR 目标相关的不等式约束
    4. build inequality for CVaR opt(构建 CVaR 优化的不等式约束)
    .....其它,看不懂....
    """
    def buildIneqConstr(self):
        # The inequality constraint is Fz<=b (不等式约束的形式)
        # Let's start by computing the submatrix of F relates with the state(首先构建与状态相关的子矩阵)
        Nc = self.Fx.shape[0]+1                     # tree中每个状态节点Nc个约束
        slackweight_x = np.zeros(self.totalx*Nc)    # tree上所有状态节点松弛权重.size:(totalx*(Fx.shape[0]+1), 1)

        bdim = self.branchdim                       # 整个tree中branch的数量
        nslack = slackweight_x.shape[0]             # 值为 self.totalx*Nc
        offset = self.totalx*self.n+self.totalu*self.d          # totalx*n + totalu*d
        Fxtot = np.zeros([Nc*self.totalx, self.totalx*self.n])  # ineq约束的系数矩阵,shape:(totalx*Nc, totalx*n)
        bxtot = np.zeros(Nc*self.totalx)                        # ineq约束的右侧向量,shape:(totalx*Nc, 1)
        # 构建tree上所有节点状态x的Ineq约束: 1.Fxtot * x_tot <= bxtot(如果有S,就要考虑在Fxtot中); 2. self.slackweight
        if self.S is None:
            for branch in self.ndx:
                l = branch.utraj.shape[0]
                # tree中每个状态节点的Ineq约束的系数矩阵,对应Fxtot中一个(Nc,n)的子矩阵; 节点Ineq约束的右侧向量,对应bxtot中一个(Nc, 1)的子向量
                # 显然每个状态节点有Nc(1+n)共5个约束:-dh*x<=h (1个); Fx*x<=bx (4个);
                for i in range(0,l):
                    # safe function h的线性化形式 h(shape:标量),与偏导 dh(shape:(1, n))
                    h,dh = self.predictiveModel.col_eval(branch.xtraj[i], branch.ztraj[i])
                    idx = self.ndx[branch]+i    # idx:当前状态节点在整个tree中的索引
                    Fxtot[idx*Nc:(idx+1)*Nc, idx*self.n:(idx+1)*self.n] = np.vstack((-dh, self.Fx))
                    bxtot[idx*Nc:(idx+1)*Nc] = np.append(h, self.bx)
                    slackweight_x[idx*Nc:(idx+1)*Nc] = branch.w # 同一个branch上的所有节点的松弛权重相同
        else:
            for branch in self.ndx:
                l = branch.utraj.shape[0]
                for i in range(0,l):
                    h,dh = self.predictiveModel.col_eval(branch.xtraj[i],branch.ztraj[i])
                    idx = self.ndx[branch]+i
                    Fxtot[idx*Nc:(idx+1)*Nc, idx*self.n:(idx+1)*self.n] = np.vstack((-dh, self.Fx@self.S))
                    bxtot[idx*Nc:(idx+1)*Nc] = np.append(h, self.bx)
                    slackweight_x[idx*Nc:(idx+1)*Nc] = branch.w

        self.slackweight = slackweight_x    # size:(totalx*Nc, 1)
        # 构建tree上所有节点输入u的Ineq约束: Futot * u_tot <= butot
        # Let's start by computing the submatrix of F relates with the input
        rep_b = [self.Fu] * (self.totalu)   # Fu是(4*2),重复self.totalu次,形成列表
        Futot = linalg.block_diag(*rep_b)   # 拼接成一个块对角矩阵,shape:(totalu*4, totalu*2)
        butot = np.tile(np.squeeze(self.bu), self.totalu)  # bu是(4*1),重复self.totalu次,形成一个向量,shape:(totalu*4, 1)

        # 构建tree上所有branch与 CVaR 目标相关的不等式约束
        Frisk = np.zeros([bdim*(2*self.m+1), bdim*(self.m*2+2)])
        Frisk[0:bdim, 0:bdim] = -np.eye(bdim)
        Frisk[bdim:, bdim*2:bdim*(2+2*self.m)] = -np.eye(2*bdim*self.m)

        # 合并所有约束
        F_hard = linalg.block_diag(Fxtot, Futot, Frisk)

        ## add slack(添加松弛变量)
        nc_x = Fxtot.shape[0] # add slack only for state constraints
        # Fist add add slack to existing constraints, the last variable is the total cost
        addSlack = np.zeros((F_hard.shape[0], nc_x+1))
        addSlack[0:nc_x, 0:nc_x] = -np.eye(nc_x)
        # Now constraint slacks >= 0
        I = - np.eye(nc_x); Zeros = np.zeros((nc_x, F_hard.shape[1]))
        Positivity = np.hstack((Zeros, I, np.zeros([nc_x,1])))
        Fl = np.vstack(( np.hstack((F_hard, addSlack)) , Positivity))
        bl = np.hstack((bxtot, butot, np.zeros(Frisk.shape[0]+nc_x)))

        # build inequality for CVaR opt(构建 CVaR 优化的不等式约束)
        #xQx+uRu<=-sigma-mup+mum-rho_child

        Fq = np.empty((0, offset+bdim*(self.m*2+2)+nslack+1))
        bq = np.empty(0)
        dims = {'q':[]}
        if self.S is None:
            W1 = self.Wx    # self.Wx 是对状态权重 Q 的 Cholesky 分解,用于SOCP中的锥约束
        else:
            W1 = self.Wx@self.S
        Jcons = np.dot(np.dot(self.xRef,self.Q),self.xRef)  # 原QP中的: xRef'*Q*xRef
        for branch in self.branchidx:
            idx = self.branchidx[branch]
            for i in range(0,self.m):
                child = branch.children[i]
                nx = child.xtraj.shape[0]
                nu = child.utraj.shape[0]
                ndx = self.ndx[child]
                ndu = self.ndu[child]
                F1 = np.zeros(offset+bdim*(self.m*2+2)+nslack+1)
                F1[offset+bdim+idx] = 1.0
                F1[offset+bdim*2+idx+i] = 1.0
                F1[offset+bdim*(2+self.m)+idx+i] = -1.0
                if child.depth<self.NB:
                    F1[offset+self.branchidx[child]] = 1.0

                F2 = np.zeros((nx*self.n+nu*self.d,offset+bdim*(self.m*2+2)+nslack+1))
                for j in range(0,nx):
                    F2[j*self.n:(j+1)*self.n,(ndx+j)*self.n:(ndx+j+1)*self.n] = -2*W1
                    F1[(ndx+j)*self.n:(ndx+j+1)*self.n] = -2*np.dot(self.xRef,self.Q)
                    F1[offset+bdim*(self.m*2+2)+(ndx+j)*Nc:offset+bdim*(self.m*2+2)+(ndx+j+1)*Nc] = self.Qslack[1]*np.ones(Nc)
                for j in range(0,nu):
                    F2[nx*self.n+j*self.d:nx*self.n+(j+1)*self.d,self.totalx*self.n+(ndu+j)*self.d:self.totalx*self.n+(ndu+j+1)*self.d] = -2*self.Wu
                F3 = -F1.copy()
                Fqi = np.vstack((F1,F2,F3))
                bqi = np.hstack((1-Jcons*nx, np.zeros(F2.shape[0]), 1+Jcons*nx))
                Fq = np.vstack((Fq,Fqi))
                bq = np.append(bq,bqi)
                dims['q'].append(bqi.shape[0])

        # add the last socp constraint regarding the total cost: J>=\rho_0+u_0^T R u_0(对应论文式(10b))
        F1 = np.zeros(offset+bdim*(self.m*2+2)+nslack+1)
        idx = self.branchidx[self.BT]
        F1[-1] = -1
        F1[offset+idx] = 1
        F1[offset+bdim*(self.m*2+2):offset+bdim*(self.m*2+2)+Nc] = self.Qslack[1]*np.ones(Nc)

        F2 = np.zeros([self.d,offset+bdim*(self.m*2+2)+nslack+1])
        uidx = self.totalx*self.n+self.ndu[self.BT]*self.d
        F2[:,uidx:uidx+self.d] = -2*self.Wu
        F3 = -F1.copy()
        Fqi = np.vstack((F1,F2,F3))
        bqi = np.hstack((1,np.zeros(F2.shape[0]),1))
        Fq = np.vstack((Fq,Fqi))
        bq = np.append(bq,bqi)
        dims['q'].append(bqi.shape[0])

        # 合并所有约束
        dims['l'] = Fl.shape[0]
        self.F = np.vstack((Fl,Fq))
        self.b = np.append(bl,bq)
        self.dims = dims


    def updateIneqConstr(self):
        Nc = self.Fx.shape[0]+1
        if self.S is None:
            W1 = self.Wx        # self.Wx 是对状态权重 Q 的 Cholesky 分解,用于SOCP中的锥约束
        else:
            W1 = self.Wx@self.S
        Jcons = np.dot(np.dot(self.xRef,self.Q),self.xRef)
        counter = self.dims['l']
        for branch in self.branchidx:
            idx = self.branchidx[branch]
            for i in range(0,self.m):
                child = branch.children[i]
                nx = child.xtraj.shape[0]
                nu = child.utraj.shape[0]
                ndx = self.ndx[child]
                ndu = self.ndu[child]

                for j in range(0,nx):
                    self.F[counter+1+j*self.n:counter+1+(j+1)*self.n,(ndx+j)*self.n:(ndx+j+1)*self.n] = -2*W1
                    self.F[counter,(ndx+j)*self.n:(ndx+j+1)*self.n] = -2*np.dot(self.xRef,self.Q)
                    self.F[counter+1+nx*self.n+nu*self.d,(ndx+j)*self.n:(ndx+j+1)*self.n] = 2*np.dot(self.xRef,self.Q)

                counter = counter+2+nx*self.n+nu*self.d
        if self.S is None:
            for branch in self.ndx:
                l = branch.utraj.shape[0]
                for i in range(0,l):
                    h,dh = self.predictiveModel.col_eval(branch.xtraj[i],branch.ztraj[i])
                    idx = self.ndx[branch]+i
                    self.F[idx*Nc,idx*self.n:(idx+1)*self.n] = -dh
                    self.b[idx*Nc] = h
                    self.slackweight[idx*Nc:(idx+1)*Nc] = branch.w
        else:
            for branch in self.ndx:
                l = branch.utraj.shape[0]
                for i in range(0,l):
                    h,dh = self.predictiveModel.col_eval(branch.xtraj[i],branch.ztraj[i])
                    dh[0] = np.sign(dh[0])*max(0.1,abs(dh[0]))
                    # if h+dh@branch.xtraj[i]<-0.5:
                    #     pdb.set_trace()
                    idx = self.ndx[branch]+i
                    self.F[idx*Nc:(idx+1)*Nc,idx*self.n:(idx+1)*self.n] = np.vstack((-dh,self.Fx@self.S))
                    self.b[idx*Nc:(idx+1)*Nc] = np.append(h,self.bx)
                    self.slackweight[idx*Nc:(idx+1)*Nc] = branch.w


    """
    输入: x ego当前状态; z obs当前状态; xRef ego目标状态; S 状态变换矩阵; Fx,bx State constraint Fx * x <= bx;
    功能: solve是整个BranchMPC_CVaR算法的核心。初始化场景树,更新场景树,构建约束,使用 ECOS 求解器求解 SOCP(二阶锥规划)问题,提取解。
    """
    def solve(self, x, z, xRef = None,S = None, Fx = None, bx = None):
        """Computes control action
        Arguments:
            x: ego vehicle state
            z: uncontrolled vehicle state
            xRef: desired state
            S: state transformation matrix (if different from identity)
        """
        # If LTV active --> identify system model
        if not xRef is None:
            self.xRef = xRef
        self.S = S

        if not Fx is None:
            self.Fx = Fx
        if not bx is None:
            self.bx = bx

        # 如果场景树未初始化,则构建场景树与不等式约束条件;否则更新场景树与不等式约束条件。
        if self.BT is None:
            self.inittree(x,z)      # inittree中scenario tree是最终结果;而trajectory tree只是生成了对应结构,并不是最终结果.
            self.buildIneqConstr()
        else:
            self.updatetree(x,z)    #根据ego,obs当前时间步新的状态与上个时间步SOCP优化结果优化的结果self.uLin,对树中各个branch进行更新.
            self.updateIneqConstr()

        self.buildEqConstr()    # 构建等式约束条件

        # 构建SOCP(二阶锥规划)问题的参数,并使用 ECOS 求解器求解
        self.q_FTOCP = np.zeros(self.F.shape[1])    # 目标函数的线性部分，最后一个元素为 1，表示最小化总成本。
        self.q_FTOCP[-1] = 1
        self.F_FTOCP = sparse.csc_matrix(self.F)    # F_FTOCP 和 b_FTOCP: 不等式约束矩阵和右侧向量
        self.b_FTOCP = self.b
        self.G_FTOCP = sparse.csc_matrix(self.G)    # 等式约束矩阵
        self.E_FTOCP = self.E       # E_FTOCP 和 L_FTOCP: 等式约束的右侧向量。
        self.L_FTOCP = self.L

        # Solve QP
        startTimer = datetime.datetime.now()
        # np.add(np.dot(self.E_FTOCP,x),self.L_FTOCP): 计算等式约束的右侧向量。
        self.ecos_solve_socp(self.q_FTOCP, self.F_FTOCP, self.b_FTOCP, self.dims, self.G_FTOCP, np.add(np.dot(self.E_FTOCP,x),self.L_FTOCP))
        self.unpackSolution()   # 解析优化结果
        endTimer = datetime.datetime.now(); deltaTimer = endTimer - startTimer
        self.solverTime = deltaTimer
        # print("Solver Time: ", self.solverTime.total_seconds(), " seconds.")


        # update applied input
        self.OldInput = self.uPred[0,:]
        self.timeStep += 1


    # 用于从优化的解中提取预测的状态序列和输入序列
    def unpackSolution(self):
        # Extract predicted state and predicted input trajectories
        if self.feasible:
            # (0 ~ self.totalx*self.n)是 xPred,需要reshape维度为(-1,self.n); 
            self.xPred = np.squeeze(np.transpose(np.reshape((self.Solution[np.arange(self.totalx*self.n)]),(-1,self.n)))).T
            # (self.totalx*self.n ~ self.totalx*self.n+self.totalu*self.d)是 uPred,需要reshape维度为(-1,self.d);
            self.uPred = np.squeeze(np.transpose(np.reshape((self.Solution[self.totalx*self.n+np.arange(self.totalu*self.d)]),(-1, self.d)))).T

            self.xLin = self.xPred  # self.xPred 从优化结果中提取,后转换为(n,totalx)的二维数组,每列表示一个状态。
            self.uLin = self.uPred  # self.uPred 从优化结果中提取,后转换为(d,totalu)的二维数组,每列表示一个输入。
            self.uLin = np.vstack((self.uLin, self.uLin[-1]))   #在 uLin 的末尾添加最后一个输入，以确保输入序列长度与状态序列长度一致。

    def BT2array(self):
        ztraj = []
        xtraj = []
        utraj = []
        branch_w = []
        q = [self.BT]
        while (len(q)>0):
            curr = q.pop(0)
            for child in curr.children:
                branch_w.append(child.w)
                ztraj.append(np.vstack((curr.ztraj[-1],child.ztraj)))
                xtraj.append(np.vstack((curr.xtraj[-1],child.xtraj)))
                utraj.append(np.vstack((curr.utraj[-1],child.utraj)))
                q.append(child)
        return xtraj,ztraj,utraj, branch_w

    # 使用 ECOS 求解器求解 SOCP(二阶锥规划)问题,并返回优化结果给 self.Solution
    def ecos_solve_socp(self, q, G, h, dims, Ae=None, be=None, initvals=None):
        """
        Solve a SOCP using ECOS:
        min cx
        s.t. G x <= h (conic constraints)
             Ae x = be
        """
        G = sparse.csc_matrix(G)

        if not Ae is None:
            A = sparse.csc_matrix(Ae)
            sol = ecos.solve(q, G, h, dims, Ae, be,verbose=False)
        else:
            sol = ecos.solve(q, G, h, dims,verbose=False)


        if sol['info']['exitFlag'] >= 0:
            self.feasible = 1
        else:
            self.feasible = 0

        """
        solution的存储结构:
        1. 一维数组,存储了对整个tree中所有branch中的所有点的x,u的优化结果,具体存储数据如下:
        2. (0 ~ self.totalx*self.n)是xPred,需要reshape维度为(-1,self.n); 
        3. (self.totalx*self.n ~ self.totalx*self.n+self.totalu*self.d)是uPred,需要reshape维度为(-1,self.d);
        """
        self.Solution = sol['x']    
