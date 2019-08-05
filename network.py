import random
import numpy as np
import scipy.linalg as LA
"""scipy linalg inversion seems to be more stable and has better accuracy"""
import collections

tempstorage={}

logger = collections.deque()
def logging(description, item):
    logger.append(item)
    with open("results.txt","a+") as file:
        file.write(description)
        file.write(str(item))
    if len(logger)>50:
        logger.popleft()
    return None

"""
Scattering network roadmap:
1. Scattering matrix and contractions, keeping diag blocks optimal
2. Regularizing things
"""


class chiral_network_layer:
    def combine_scattering_matrices(blocks1, blocks2):
        """for converted matrices, returns unitary matrix
        C(U_s & U_o)= C(U_o)@C(U_s)
        where C is the deconversion algorithm
        self => 2
        other => 1
        """
        a1 = blocks1[0][0]
        b1 = blocks1[0][1]
        c1 = blocks1[1][0]
        d1 = blocks1[1][1]
        a2 = blocks2[0][0]
        b2 = blocks2[0][1]
        c2 = blocks2[1][0]
        d2 = blocks2[1][1]
        m, n = b2.shape
        prefix2 = d2@LA.inv(np.identity(n)-c1@b2)
        prefix1 = a1@LA.inv(np.identity(m)-b2@c1)
        return [[prefix1@a2,b1+prefix1@(b2@d1)],[c2+prefix2@(c1@a2),prefix2@d1]]
    def blockconversion(blocks):
        """
        Converts matrices pseudo-unitary under R/L odd signature metric to its "dual" unitary matrices and vice-versa.
        This basically converts scattering matrices to transfer matrices and vice-versa
        """
        try:
            a = blocks[0][0]
            b = blocks[0][1]
            c = blocks[1][0]
            d = blocks[1][1]
            dinv = LA.inv(d)
            bdi = b@dinv
            newa = a - bdi@c
            newb = bdi
            newc = -dinv@c
            newd = dinv
        except LA.LinAlgError:
            print(blocks)
        return [[newa, newb],[newc, newd]]
    def __init__(self, L, R, dtype = np.dtype(np.complex128)):
        self.dtype = dtype
        self.R = R
        self.L = L
    def transfer_blocks(self, N=1):
        """yields the transfer matrix of a single layer, in blocks"""
        raise NotImplementedError("This is an interface")
    def scattering_blocks(self, N=1, collection=4):
        """yields the scattering matrix of a single layer, in blocks"""
        i = 0
        combined_block = np.identity(self.L+self.R)
        for blocks in self.transfer_blocks(N):
            yield chiral_network_layer.blockconversion(blocks)
    def transfer(self, N=1):
        """yields the transfer matrix of a single layer"""
        for blocks in self.transfer_blocks(N):
            yield np.block(blocks)
    def scattering(self, N=1, collection=4):
        for blocks in self.scattering_blocks(N, collection):
            yield np.block(blocks)

class clean_flat_layer(chiral_network_layer):
    def __init__(self, N, delta = 0.0, norm = 1.0, periodic=False, dtype = np.dtype(np.complex128)):
        assert N%2 == 0 or not periodic, "periodicity only possible for even network widths"
        L = N//2
        R = N - L
        chiral_network_layer.__init__(self, L, R,  dtype = dtype)
        self.delta = delta
        self.norm = norm
        self.periodic = periodic
        """calculation of important constants"""
        sqrt2 = np.sqrt(2, dtype = self.dtype)
        delta2 = self.delta/2
        tempa = np.sqrt(1-2*delta2**2, dtype = self.dtype)/sqrt2
        self.theta1 = np.arctanh((tempa-delta2)*self.norm, dtype = self.dtype)
        self.theta2 = -np.arctanh((tempa+delta2)*self.norm, dtype = self.dtype)
        self.s1 = np.sinh(self.theta1, dtype = self.dtype)
        self.s2 = np.sinh(self.theta2, dtype = self.dtype)
        self.c1 = np.cosh(self.theta1, dtype = self.dtype)
        self.c2 = np.cosh(self.theta2, dtype = self.dtype)
        """base building blocks"""
        self.rr1 = np.identity(R, dtype = self.dtype)
        self.rl1 = np.zeros(shape=(R,L), dtype = self.dtype)
        self.lr1 = np.zeros(shape=(L,R), dtype = self.dtype)
        self.ll1 = np.identity(L, dtype = self.dtype)
        for i in range(min(self.L, self.R)):
            self.ll1[i,i]=self.c1
            self.rr1[i,i]=self.c1
            self.lr1[i,i]=self.s1
            self.rl1[i,i]=self.s1
        self.rr2 = np.identity(R, dtype = self.dtype)
        self.rl2 = np.zeros(shape=(R,L), dtype = self.dtype)
        self.lr2 = np.zeros(shape=(L,R), dtype = self.dtype)
        self.ll2 = np.identity(L, dtype = self.dtype)
        if not self.periodic:
            for i in range(min(self.L, self.R-1)):
                self.ll2[i,i]=self.c2
                self.rr2[i+1,i+1]=self.c2
                self.lr2[i,i+1]=self.s2
                self.rl2[i+1,i]=self.s2
        else:
            for i in range(self.L):
                self.ll2[i,i]=self.c2
                self.rr2[(i+1)%self.L,(i+1)%self.L]=self.c2
                self.lr2[i,(i+1)%self.L]=self.s2
                self.rl2[(i+1)%self.L,i]=self.s2
        self.rr = self.rr1@self.rr2+self.rl1@self.lr2
        self.rl = self.rl1@self.ll2+self.rr1@self.rl2
        self.lr = self.lr1@self.rr2+self.ll1@self.lr2
        self.ll = self.ll1@self.ll2+self.lr1@self.rl2
    def transfer_blocks(self, N=1):
        for i in range(N):
            #print(i)
            yield [[self.rr, self.rl],[self.lr, self.ll]]
    def scattering_blocks(self, N=1):
        scat = chiral_network_layer.blockconversion([[self.rr, self.rl],[self.lr, self.ll]])
        for i in range(N):
            #print(i)
            yield scat
    def transfer(self, N=1):
        """yields the transfer matrix of a single layer"""
        result = np.block([[self.rr, self.rl],[self.lr, self.ll]])
        for i in range(N):
            #print(i)
            yield result
    def scattering(self, N=1):
        result = np.block(chiral_network_layer.blockconversion([[self.rr, self.rl],[self.lr, self.ll]]))
        for i in range(N):
            #print(i)
            yield result

class noisy_flat_layer(chiral_network_layer):
    def __init__(self, N, delta = 0.0, norm = 1.0, periodic=False, dtype = np.dtype(np.complex128), W=0, noisetype="paired_flip"):
        self.W = W
        self.noisetype = noisetype
        clean_flat_layer.__init__(self, N, delta, norm, periodic, dtype)
    def transfer_blocks(self, N=1):
        W = self.W
        if self.noisetype == "paired_flip":
            for i in range(N):
                Z1 = np.diag([-1 if np.random.random()<W else 1 for i in range(self.R)])
                Z2 = np.diag([-1 if np.random.random()<W else 1 for i in range(self.R)])
                Zrl1 = Z1@self.rl1
                lrZ1 = self.lr1@Z1
                Zrl2 = Z2@self.rl2
                lrZ2 = self.lr2@Z2
                rr = self.rr1@self.rr2+Zrl1@lrZ2
                rl = Zrl1@self.ll2+self.rr1@Zrl2
                lr = lrZ1@self.rr2+self.ll1@lrZ2
                ll = self.ll1@self.ll2+lrZ1@Zrl2
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "iso_flip":
            for i in range(N):
                Z1r = np.array([-1 if np.random.random()<W else 1 for i in range(self.R)])
                Z1l = np.array([-1 if np.random.random()<W else 1 for i in range(self.L)])
                Z2r = np.array([-1 if np.random.random()<W else 1 for i in range(self.R)])
                Z2l = np.array([-1 if np.random.random()<W else 1 for i in range(self.L)])
                rr = (self.rr1*Z1r)@self.rr2*Z2r+(self.rl1*Z1l)@self.lr2*Z2r
                rl = (self.rl1*Z1l)@self.ll2*Z2l+(self.rr1*Z1r)@self.rl2*Z2l
                lr = (self.lr1*Z1r)@self.rr2*Z2r+(self.ll1*Z1l)@self.lr2*Z2r
                ll = (self.ll1*Z1l)@self.ll2*Z2l+(self.lr1*Z1r)@self.rl2*Z2l
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "chiral_flip":
            for i in range(N):
                Z1r = np.array([-1 if np.random.random()<W else 1 for i in range(self.R)])
                Z2r = np.array([-1 if np.random.random()<W else 1 for i in range(self.R)])
                rr = (self.rr1*Z1r)@self.rr2*Z2r+self.rl1@self.lr2*Z2r
                rl = self.rl1@self.ll2+(self.rr1*Z1r)@self.rl2
                lr = (self.lr1*Z1r)@self.rr2*Z2r+self.ll1@self.lr2*Z2r
                ll = self.ll1@self.ll2+(self.lr1*Z1r)@self.rl2
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "chiral_flip_dbl":
            rrc = self.rr1@self.rr2+self.rl1@self.lr2
            rlc = self.rl1@self.ll2+self.rr1@self.rl2
            lrc = self.lr1@self.rr2+self.ll1@self.lr2
            llc = self.ll1@self.ll2+self.lr1@self.rl2
            for i in range(N):
                Zr = np.array([-1 if np.random.random()<W else 1 for i in range(self.R)])
                rr = rrc*Zr
                rl = rlc
                lr = lrc*Zr
                ll = llc
                yield [[rr, rl],[lr, ll]]
        W = 2*self.W
        if self.noisetype == "paired_rotation":
            for i in range(N):
                Z1 = np.diag([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Z2 = np.diag([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zrl1 = Z1@self.rl1
                lrZ1 = self.lr1@Z1
                Zrl2 = Z2@self.rl2
                lrZ2 = self.lr2@Z2
                rr = self.rr1@self.rr2+Zrl1@lrZ2
                rl = Zrl1@self.ll2+self.rr1@Zrl2
                lr = lrZ1@self.rr2+self.ll1@lrZ2
                ll = self.ll1@self.ll2+lrZ1@Zrl2
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "iso_rotation":
            for i in range(N):
                Z1r = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Z1l = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.L)])
                Z2r = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Z2l = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.L)])
                rr = (self.rr1*Z1r)@self.rr2*Z2r+(self.rl1*Z1l)@self.lr2*Z2r
                rl = (self.rl1*Z1l)@self.ll2*Z2l+(self.rr1*Z1r)@self.rl2*Z2l
                lr = (self.lr1*Z1r)@self.rr2*Z2r+(self.ll1*Z1l)@self.lr2*Z2r
                ll = (self.ll1*Z1l)@self.ll2*Z2l+(self.lr1*Z1r)@self.rl2*Z2l
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "chiral_rotation":
            for i in range(N):
                Z1r = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Z2r = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                rr = (self.rr1*Z1r)@self.rr2*Z2r+self.rl1@self.lr2*Z2r
                rl = self.rl1@self.ll2+(self.rr1*Z1r)@self.rl2
                lr = (self.lr1*Z1r)@self.rr2*Z2r+self.ll1@self.lr2*Z2r
                ll = self.ll1@self.ll2+(self.lr1*Z1r)@self.rl2
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "chiral_rot_dbl":
            rrc = self.rr1@self.rr2+self.rl1@self.lr2
            rlc = self.rl1@self.ll2+self.rr1@self.rl2
            lrc = self.lr1@self.rr2+self.ll1@self.lr2
            llc = self.ll1@self.ll2+self.lr1@self.rl2
            for i in range(N):
                Zr = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                rr = rrc*Zr
                rl = rlc
                lr = lrc*Zr
                ll = llc
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "iso_rot_dbl":
            rrc = self.rr1@self.rr2+self.rl1@self.lr2
            rlc = self.rl1@self.ll2+self.rr1@self.rl2
            lrc = self.lr1@self.rr2+self.ll1@self.lr2
            llc = self.ll1@self.ll2+self.lr1@self.rl2
            for i in range(N):
                Zr = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zl = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.L)])
                rr = rrc*Zr
                rl = rlc*Zl
                lr = lrc*Zr
                ll = llc*Zl
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "Ax":
            rrc = self.rr1@self.rr2+self.rl1@self.lr2
            rlc = self.rl1@self.ll2+self.rr1@self.rl2
            lrc = self.lr1@self.rr2+self.ll1@self.lr2
            llc = self.ll1@self.ll2+self.lr1@self.rl2
            for i in range(N):
                expst = np.array([(np.random.random()-0.5)*W for i in range(self.R)])
                Zr = np.exp(1j*expst)
                Zl = np.exp(-1j*expst)
                I1 = (self.rl1*Zl)
                I2 = (self.ll1*Zl)
                rr = (self.rr1@self.rr2+I1@self.lr2)*Zr
                rl = I1@self.ll2+self.rr1@self.rl2
                lr = (self.lr1@self.rr2+I2@self.lr2)*Zr
                ll = I2@self.ll2+self.lr1@self.rl2
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "V":
            rrc = self.rr1@self.rr2+self.rl1@self.lr2
            rlc = self.rl1@self.ll2+self.rr1@self.rl2
            lrc = self.lr1@self.rr2+self.ll1@self.lr2
            llc = self.ll1@self.ll2+self.lr1@self.rl2
            for i in range(N):
                expst = np.array([(np.random.random()-0.5)*W for i in range(self.R)])
                Zr = np.exp(1j*expst)
                Zl = Zr
                I1 = (self.rl1*Zl)
                I2 = (self.ll1*Zl)
                rr = (self.rr1@self.rr2+I1@self.lr2)*Zr
                rl = I1@self.ll2+self.rr1@self.rl2
                lr = (self.lr1@self.rr2+I2@self.lr2)*Zr
                ll = I2@self.ll2+self.lr1@self.rl2
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "p12":
            rrc = self.rr1@self.rr2+self.rl1@self.lr2
            rlc = self.rl1@self.ll2+self.rr1@self.rl2
            lrc = self.lr1@self.rr2+self.ll1@self.lr2
            llc = self.ll1@self.ll2+self.lr1@self.rl2
            for i in range(N):
                Z = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                rrc = (self.rr*Z)@(self.rr2*Z)+self.rl1@(self.lr2*Z)
                rlc = self.rl1@self.ll2+(self.rr1*Z)@self.rl2
                lrc = (self.lr1*Z)@(self.rr2*Z)+self.ll1@(self.lr2*Z)
                llc = self.ll1@self.ll2+(self.lr1*Z)@self.rl2
                rr = rrc
                rl = rlc
                lr = lrc
                ll = llc
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "phi2":
            print("phi2")
            rrc = self.rr1@self.rr2+self.rl1@self.lr2
            rlc = self.rl1@self.ll2+self.rr1@self.rl2
            lrc = self.lr1@self.rr2+self.ll1@self.lr2
            llc = self.ll1@self.ll2+self.lr1@self.rl2
            for i in range(N):
                Z = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                rrc = (self.rr*Z)@self.rr2+self.rl1@self.lr2
                rlc = self.rl1@self.ll2+(self.rr1*Z)@self.rl2
                lrc = (self.lr1*Z)@self.rr2+self.ll1@self.lr2
                llc = self.ll1@self.ll2+(self.lr1*Z)@self.rl2
                rr = rrc
                rl = rlc
                lr = lrc
                ll = llc
                yield [[rr, rl],[lr, ll]]


class clean_uniform_layer(chiral_network_layer):
    def __init__(self, N, params={}, periodic=True, dtype = np.dtype(np.complex128)):
        assert N%2 == 0, "model only workable for even N"
        L = N//2
        R = N - L
        chiral_network_layer.__init__(self, L, R,  dtype = dtype)
        self.params = {'l':1, 'w':1, 'e':1, 'Ax':0, 'Ay1':0, 'Ay2':0, 'V':0, 'mu':0, 'v':1, 'offset':True}
        for key in params:
            self.params[key]=params[key]
        self.size = L
        self.l = self.params['l']
        self.w = self.params['w']
        self.e = self.params['e']
        self.v = self.params['v']
        self.mu = self.params['mu']
        self.Ax = self.params['Ax']
        self.Ay1 = self.params['Ay1']
        self.Ay2 = self.params['Ay2']
        self.V = self.params['V']
        self.offset = self.params['offset']
        CYC = np.roll(np.identity(self.size), 1, axis=1)
        IDN = np.identity(self.size)
        if self.offset == True or self.offset== "half":
            pre_exp = [[(self.e*self.Ax-self.V/self.v)*IDN,
                        0.5j*self.e*self.Ay1*IDN+0.5j*self.e*self.Ay2*CYC+(-self.mu/self.v/2)*(IDN+CYC)+(IDN-CYC)/self.w],
                       [0.5j*self.e*self.Ay1*IDN+0.5j*self.e*self.Ay2*CYC.T+(self.mu/self.v/2)*(IDN+CYC.T)-(IDN-CYC.T)/self.w,
                        (self.e*self.Ax+self.V/self.v)*IDN]]
        elif self.offset == "derivative":
            pre_exp = [[(self.e*self.Ax-self.V/self.v)*IDN,
                        (1j*self.e*self.Ay1-self.mu/self.v)*IDN+(IDN-CYC)/self.w],
                       [(1j*self.e*self.Ay1+self.mu/self.v)*IDN-(IDN-CYC.T)/self.w,
                        (self.e*self.Ax+self.V/self.v)*IDN]]
        elif self.offset == "direct":
            pre_exp = [[(self.e*self.Ax-self.V/self.v)*IDN,
                        (1j*self.e*self.Ay1-self.mu/self.v)*IDN+(CYC.T-CYC)/self.w],
                       [(1j*self.e*self.Ay1+self.mu/self.v)*IDN+(CYC.T-CYC)/self.w,
                        (self.e*self.Ax+self.V/self.v)*IDN]]
        else:
            raise NotImplementedError()
        print("flag")
        tempstorage['pre_exp'] = pre_exp
        self.transfer = LA.expm(-1j*self.l*np.block(pre_exp))
        if np.amax(self.transfer)>5000:
            print("warning", np.max(self.transfer))
        self.t_rr = self.transfer[:self.R,:self.R]
        self.t_rl = self.transfer[:self.R,self.R:]
        self.t_lr = self.transfer[self.R:,:self.R]
        self.t_ll = self.transfer[self.R:,self.R:]
        self.scattering = np.block(chiral_network_layer.blockconversion([[self.t_rr, self.t_rl],[self.t_lr, self.t_ll]]))
        self.s_rr = self.scattering[:self.R,:self.R]
        self.s_rl = self.scattering[:self.R,self.R:]
        self.s_lr = self.scattering[self.R:,:self.R]
        self.s_ll = self.scattering[self.R:,self.R:]
        #print(self.scattering)
    def transfer_blocks(self, N=1):
        for i in range(N):
            yield [[self.t_rr, self.t_rl],[self.t_lr, self.t_ll]]
    def scattering_blocks(self, N=1):
        for i in range(N):
            yield [[self.s_rr, self.s_rl],[self.s_lr, self.s_ll]]
    def transfer(self, N=1):
        for i in range(N):
            yield self.transfer
    def scattering(self, N=1):
        for i in range(N):
            yield self.scattering

"""only supports Ax and V noise; Ay noise depends on noise depth"""
class noisy_uniform_layer(chiral_network_layer):
    def cyc(self, array):
        return np.roll(np.diag(array), 1, axis=1)
    def matgen(self, **new_params):
        params = {'l':1, 'w':1, 'e':1, 'Ax':np.zeros(self.size), 'Ay1':np.zeros(self.size), 'Ay2':np.zeros(self.size), 'V':np.zeros(self.size), 'mu':0, 'v':1}
        for i in new_params:
            params[i]=new_params[i]
        l = params['l']
        w = params['w']
        e = params['e']
        v = params['v']
        mu = params['mu']
        Ax = params['Ax']#array
        Ay1 = params['Ay1']#array
        Ay2 = params['Ay2']
        V = params['V']#array
        CYC = np.roll(np.identity(self.size), 1, axis=1)
        IDN = np.identity(self.size)
        t1 = self.cyc(Ay2)
        roll = (e*Ax+V/v)
        roll = (roll+np.roll(roll, 1))/2
        pre_exp = [[np.diag(e*Ax-V/v),
                    0.5j*e*np.diag(Ay1)+0.5j*e*t1-(mu/v/2)*(IDN+CYC)+(IDN-CYC)/w],
                   [0.5j*e*np.diag(Ay1)+0.5j*e*t1.T+(mu/v/2)*(IDN+CYC.T)-(IDN-CYC.T)/w,
                    np.diag(roll)]]
        transfer = LA.expm(-1j*l*np.block(pre_exp))
        if np.amax(transfer)>5000:
            print("warning", np.max(transfer))
        t_rr = transfer[:self.R,:self.R]
        t_rl = transfer[:self.R,self.R:]
        t_lr = transfer[self.R:,:self.R]
        t_ll = transfer[self.R:,self.R:]
        temp = np.block(chiral_network_layer.blockconversion([[t_rr, t_rl],[t_lr, t_ll]]))
        print(np.around(temp.conj().T@temp-np.identity(2*self.size), decimals=5))
        return chiral_network_layer.blockconversion([[t_rr, t_rl],[t_lr, t_ll]])
    def two_blockconv(self,matx):
        """
        Converts matrices pseudo-unitary under R/L odd signature metric to its "dual" unitary matrices and vice-versa.
        This basically converts scattering matrices to transfer matrices and vice-versa
        """
        a = matx[0][0]
        b = matx[0][1]
        c = matx[1][0]
        d = matx[1][1]
        newd = 1/d
        bdi = b*newd
        newa = a - bdi*c
        newb = bdi
        newc = -newd*c
        return [[newa, newb],[newc, newd]]
    def __init__(self, N, params={}, periodic=True, dtype = np.dtype(np.complex128), noisetype="default"):
        self.noisetype = noisetype
        if 'noise_strength' in params:
            self.noise_strength = params['noise_strength']
        else:
            print("no noise strength specified")
            self.noise_strength = 1.0
        clean_uniform_layer.__init__(self, N, params, periodic, dtype)
    def transfer_blocks(self, N=1):
        for i in range(N):
            if self.noisetype=="default":
                Zr = np.array([np.exp(2*np.pi*1j*np.random.random()) for i in range(self.R)])
            elif self.noisetype=="none":
                Zr = np.array([1 for i in range(self.R)])
            else:
                Zr = np.array([np.exp(2*np.pi*1j*np.random.random()) for i in range(self.R)])
            yield [[self.t_rr, self.t_rl*Zr],[self.t_lr, self.t_ll*Zr]]
    def scattering_blocks(self, N=1):
        if self.noisetype=="default":
            for i in range(N):
                Zl = np.array([np.exp(2*np.pi*1j*np.random.random()) for i in range(self.R)])
                yield [[self.s_rr, self.s_rl*Zl],[self.s_lr, self.s_ll*Zl]]
#        elif self.noisetype=="rot90":
#            for i in range(N):
#                k = np.random.normal(size=self.R)*self.noise_strength
#                mag = (1-k*1j)/(1 + k**2)
#                A = np.diag(mag)
#                B = np.diag(k*mag)
#                yield chiral_network_layer.combine_scattering_matrices([[self.s_rr, self.s_rl],[self.s_lr, self.s_ll]],[[A,B],[-B,A]])
        elif self.noisetype[:6]=="rotsym":
            CYC = np.roll(np.identity(self.size), 1, axis=1)
            theta = int(self.noisetype[6:])/180*np.pi
            rot45 = np.array([[np.cos(theta)-1,-np.sin(theta)*1j],[-np.sin(theta)*1j,np.cos(theta)+1]])
            def gen(toggle=False):
                nonlocal rot45, self, CYC
                noise_amps = np.random.normal(size=self.R)*self.noise_strength
                Zrr = []
                Zrl = []
                Zlr = []
                Zll = []
                for k in noise_amps:
                    transfer = LA.expm(1j*rot45*k)
                    scattering = self.two_blockconv(transfer)
                    Zrr.append(scattering[0][0])
                    Zrl.append(scattering[0][1])
                    Zlr.append(scattering[1][0])
                    Zll.append(scattering[1][1])
                if toggle:
                    return [[np.diag(Zrr),np.diag(Zrl)@CYC.T],[CYC@np.diag(Zlr),np.diag(Zll)]]
                else:
                    return [[np.diag(Zrr),np.diag(Zrl)],[np.diag(Zlr),np.diag(Zll)]]
            for i in range(N):
                block1 = gen(False)
                block2 = gen(True)
                midblocks = chiral_network_layer.combine_scattering_matrices(block1, block2)
                yield chiral_network_layer.combine_scattering_matrices([[self.s_rr, self.s_rl],[self.s_lr, self.s_ll]],midblocks)
        elif self.noisetype[:3]=="rot":
            print("rot")
            theta = int(self.noisetype[3:])/180*np.pi
            rot = np.array([[np.cos(theta)-1,-np.sin(theta)*1j],[-np.sin(theta)*1j,np.cos(theta)+1]])
            for i in range(N):
                noise_amps = np.random.normal(size=self.R)*self.noise_strength
                Zrr = []
                Zrl = []
                Zlr = []
                Zll = []
                for k in noise_amps:
                    transfer = LA.expm(1j*rot*k)
                    scattering = self.two_blockconv(transfer)
                    Zrr.append(scattering[0][0])
                    Zrl.append(scattering[0][1])
                    Zlr.append(scattering[1][0])
                    Zll.append(scattering[1][1])
                tempstorage['matrix'] = chiral_network_layer.combine_scattering_matrices([[self.s_rr, self.s_rl],[self.s_lr, self.s_ll]],[[np.diag(Zrr),np.diag(Zrl)],[np.diag(Zlr),np.diag(Zll)]])
                yield tempstorage['matrix']
        elif self.noisetype=="none":
            print("none")
            for i in range(N):
                tempstorage['matrix'] = [[self.s_rr, self.s_rl],[self.s_lr, self.s_ll]]
                yield tempstorage['matrix']
        elif self.noisetype=="precise_default":
            for i in range(N):
                Ax = np.random.normal(size=self.size)*self.noise_strength
                V = Ax
                yield self.matgen(l=self.l, w=self.w, mu=self.mu, e=self.e, v=self.v, Ax=Ax, V=V)
        elif self.noisetype[:11]=="precise_rot":
            angle = int(self.noisetype[11:])/180*np.pi
            c = np.cos(angle)
            s = np.sin(angle)
            for i in range(N):
                V = np.random.normal(size=self.size)*self.noise_strength
                Ax = V*c
                Ay1 = V*s
                Ay2 = (Ay1+np.roll(Ay1,-1))/2
                yield self.matgen(l=self.l, w=self.w, mu=self.mu, e=self.e, v=self.v, Ax=Ax, Ay1=Ay1, Ay2=Ay2, V=V)
        else:
            for i in range(N):
                Zl = np.array([np.exp(2*np.pi*1j*np.random.random()) for i in range(self.R)])
                yield [[self.s_rr, self.s_rl.T*Zl],[self.s_lr, self.s_ll.T*Zl]]
    def transfer(self, N=1):
        for i in self.transfer_blocks(N):
            yield np.block(i)
    def scattering(self, N=1):
        for i in self.scattering_blocks(N):
            yield np.block(i)

class noisy_uniform_layer_span(chiral_network_layer):
    pass

class clean_wave_layer(chiral_network_layer):
    def super_roll(self,array):
        return np.hstack([[np.roll(array,self.N - i) for i in range(self.size)]])
    def genlayer(self, params):
        Ax = params['Ax']
        Ay = params['Ay']
        V = params['V']
        v = params['v']
        W = params['W']
        mu = params['mu']
        e = params['e']
        k_0 = 2*np.pi/W
        l = params['l']
        N = self.N
        size = self.size
        counter = np.linspace(-N,N,2*N+1)
        eAx = e*Ax
        Vov = V/v
        RRcounter = -eAx-Vov
        LLcounter = -eAx+Vov
        RLdiag = (1j*k_0/v)*counter - mu/v
        LRdiag = (1j*k_0/v)*counter + mu/v
        sq = -self.super_roll(1j*e*Ay)
        pre_exp = [[self.super_roll(RRcounter),sq+np.diag(LRdiag)],[sq+np.diag(RLdiag),self.super_roll(LLcounter)]]
        #print(pre_exp)
        return LA.expm(1j*l*np.block(pre_exp))
    def __init__(self, N, params={}, periodic=True, dtype = np.dtype(np.complex128), noisetype="none"):
        L = 2*N + 1
        R = 2*N + 1
        chiral_network_layer.__init__(self, N, N,  dtype = dtype)
        self.N = N
        self.size = 2*N + 1
        self.params = {'Ax': np.array([0]*self.size), 'Ay':np.array([0]*self.size), 'V':np.array([0]*self.size), 'v':1.0, 'W':1.0, 'e':1.0, 'mu':0.0, 'l':1.0}
        for key in params:
            self.params[key]=params[key]
        self.transfer = self.genlayer(self.params)
        #print(self.transfer)
        self.t_rr = self.transfer[:self.R,:self.R]
        self.t_rl = self.transfer[:self.R,self.R:]
        self.t_lr = self.transfer[self.R:,:self.R]
        self.t_ll = self.transfer[self.R:,self.R:]
        self.scattering = np.block(chiral_network_layer.blockconversion([[self.t_rr, self.t_rl],[self.t_lr, self.t_ll]]))
        self.s_rr = self.scattering[:self.R,:self.R]
        self.s_rl = self.scattering[:self.R,self.R:]
        self.s_lr = self.scattering[self.R:,:self.R]
        self.s_ll = self.scattering[self.R:,self.R:]
    def transfer_blocks(self, N=1):
        for i in range(N):
            yield [[self.t_rr, self.t_rl],[self.t_lr, self.t_ll]]
    def scattering_blocks(self, N=1):
        for i in range(N):
            yield [[self.s_rr, self.s_rl],[self.s_lr, self.s_ll]]
    def transfer(self, N=1):
        for i in range(N):
            yield self.transfer
    def scattering(self, N=1):
        for i in range(N):
            yield self.scattering

class clean_square_layer(chiral_network_layer):
    def __init__(self, N, beta=0.0, periodic=True, dtype = np.dtype(np.complex128)):
        assert periodic, "square layer only defined for periodic conditions"
        assert N%2 == 0, "square model only defined for even widths"
        L = N//2
        R = L
        chiral_network_layer.__init__(self, L, R,  dtype = dtype)
        """calculation of important constants. random phase shifts are always outer"""
        theta = beta + np.pi/4
        self.s = np.sin(theta, dtype = self.dtype)
        self.c = np.cos(theta, dtype = self.dtype)
        phi1 = np.ones(R)
        phi2 = np.ones(R)
        phi3 = np.ones(R)
        phi4 = np.ones(R)
        M = np.diag(phi2)
        Y = np.roll(np.diag(phi4),1,axis=0)
        #A13 means layer A in 1 out 3
        self.A11 = (M+Y)*((self.s*self.c)*phi1)
        self.A33 = self.A11*phi3
        self.A13 = (M*self.s**2-Y*self.c**2)*phi3
        self.A31 = (M*self.c**2-Y*self.s**2)*phi1
        phi1 = np.ones(R)
        phi2 = np.ones(R)
        phi3 = np.ones(R)
        phi4 = np.ones(R)
        M = np.diag(phi4)
        Y = np.roll(np.diag(phi2),1,axis=1)
        #A13 means layer A in 1 out 3
        self.B11 = (M+Y)*((self.s*self.c)*phi1)
        self.B33 = self.A11*phi3
        self.B13 = (Y*self.s**2-M*self.c**2)*phi3
        self.B31 = (Y*self.c**2-M*self.s**2)*phi1
        scat1 = [[self.A11,self.A13],[self.A31,self.A33]]
        scat2 = [[self.B11,self.B13],[self.B31,self.B33]]
        self.scattering = chiral_network_layer.combine_scattering_matrices(scat1, scat2)
        ## DEBUG:
        temp = np.block(self.scattering)
        print("check unitarity", temp.conj().T@temp)
    def transfer_blocks(self, N=1):
        for blocks in self.scattering_blocks(N):
            yield chiral_network_layer.blockconversion(blocks)
    def scattering_blocks(self, N=1):
        for i in range(N):
            #print(i)
            yield self.scattering

class noisy_square_layer(chiral_network_layer):
    def __init__(self, N, beta=0.0, periodic=True, dtype = np.dtype(np.complex128), W=0.0, noisetype="chiral"):
        assert periodic, "square layer only defined for periodic conditions"
        assert N%2 == 0, "square model only defined for even widths"
        self.W = W
        self.noisetype = noisetype
        L = N//2
        R = L
        chiral_network_layer.__init__(self, L, R,  dtype = dtype)
        """calculation of important constants. random phase shifts are always outer"""
        theta = beta + np.pi/4
        self.s = np.sin(theta, dtype = self.dtype)
        self.c = np.cos(theta, dtype = self.dtype)
        self.s_q = self.s**2
        self.c_q = self.c**2
        self.sc = self.s*self.c
    def transfer_blocks(self, N=1):
        for blocks in self.scattering_blocks(N):
            yield chiral_network_layer.blockconversion(blocks)
    def scattering_blocks(self, N=1):
        noisetype = self.noisetype
        W = 2*self.W
        R = self.R
        allphis = set([])
        if noisetype not in allphis:
            """precomputes"""
            M = np.identity(R)
            Y = np.roll(np.identity(R),1,axis=0)
            self.A11 = (M+Y)*(self.s*self.c)
            self.A33 = self.A11
            self.A13 = (M*self.s**2-Y*self.c**2)
            self.A31 = (M*self.c**2-Y*self.s**2)
            M = np.identity(R)
            Y = np.roll(np.identity(R),1,axis=1)
            self.B11 = (M+Y)*(self.s*self.c)
            self.B33 = self.A11
            self.B13 = (Y*self.s**2-M*self.c**2)
            self.B31 = (Y*self.c**2-M*self.s**2)
            for i in range(N):
                """noisetype"""
                if noisetype=="chiral_rot_dbl":
                    phi1A = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi3A = np.ones(R)
                    phi1B = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi3B = np.ones(R)
                else:
                    phi1A = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi3A = np.ones(R)
                    phi1B = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi3B = np.ones(R)
                """compute"""
                scat1 = [[self.A11*phi1A,self.A13*phi3A],[self.A31*phi1A,self.A33*phi3A]]
                scat2 = [[self.B11*phi1B,self.B13*phi3B],[self.B31*phi1B,self.B33*phi3B]]
                yield chiral_network_layer.combine_scattering_matrices(scat1, scat2)
        else:
            for i in range(N):
                """generate noise"""
                if noisetype=="uniform":
                    phi1A = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi2A = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi3A = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi4A = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi1B = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi2B = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi3B = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                    phi4B = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                """compute"""
                M = np.diag(phi2A)
                Y = np.roll(np.diag(phi4A),1,axis=0)
                self.A11 = (M+Y)*((self.s*self.c)*phi1A)
                self.A33 = self.A11*phi3A
                self.A13 = (M*self.s**2-Y*self.c**2)*phi3A
                self.A31 = (M*self.c**2-Y*self.s**2)*phi1A
                M = np.diag(phi4B)
                Y = np.roll(np.diag(phi2B),1,axis=1)
                #A13 means layer A in 1 out 3
                self.B11 = (M+Y)*((self.s*self.c)*phi1B)
                self.B33 = self.A11*phi3B
                self.B13 = (Y*self.s**2-M*self.c**2)*phi3B
                self.B31 = (Y*self.c**2-M*self.s**2)*phi1B
                scat1 = [[self.A11,self.A13],[self.A31,self.A33]]
                scat2 = [[self.B11,self.B13],[self.B31,self.B33]]
                yield chiral_network_layer.combine_scattering_matrices(scat1, scat2)

class clean_square_layer_old(chiral_network_layer):
    def __init__(self, N, beta=0.0, periodic=True, dtype = np.dtype(np.complex128)):
        assert periodic, "square layer only defined for periodic conditions"
        assert N%2 == 0, "square model only defined for even widths"
        L = N//2
        R = L
        chiral_network_layer.__init__(self, L, R,  dtype = dtype)
        """calculation of important constants. random phase shifts are always outer"""
        theta = beta + np.pi/4
        self.s = np.sin(theta, dtype = self.dtype)
        self.c = np.cos(theta, dtype = self.dtype)
        Zl1 = np.ones(R)
        Zr1 = np.ones(R)
        Zl2 = np.ones(R)
        Zr2 = np.ones(R)
        cyc1 = np.roll(np.identity(R),1,axis=0)*Zr2
        match1 = np.diag(Zl2)
        self.layer1LL = cyc1*self.c**2-match1*self.s**2
        self.layer1RR = match1*self.c**2-cyc1*self.s**2
        self.layer1RL = (cyc1+match1)*self.s*self.c
        self.layer1LR = -self.layer1RL
        scat1 = [[self.layer1LR*Zl1,self.layer1RR*Zr1],[self.layer1LL*Zl1,self.layer1RL*Zr1]]
        Xl1 = np.ones(R)
        Xr1 = np.ones(R)
        Xl2 = np.ones(R)
        Xr2 = np.ones(R)
        cyc2 = np.roll(np.identity(R),1,axis=0)*Xr2
        match2 = np.diag(Xl2)
        self.layer2LL = cyc2*self.s**2-match2*self.c**2
        self.layer2RR = match2*self.s**2-cyc2*self.c**2
        self.layer2RL = (cyc2+match2)*self.s*self.c
        self.layer2LR = -self.layer2RL
        scat2 = [[self.layer2LR*Zl1,self.layer2RR*Zr1],[self.layer2LL*Zl1,self.layer2RL*Zr1]]
        self.scattering = chiral_network_layer.combine_scattering_matrices(scat1, scat2)
        """
        self.theta1 = np.arctanh(self.s)
        self.theta2 = np.arctanh(self.c)
        print(self.theta1,self.theta2)
        self.s1 = np.sinh(self.theta1, dtype = self.dtype)
        self.c1 = np.cosh(self.theta1, dtype = self.dtype)
        self.s2 = np.sinh(self.theta2, dtype = self.dtype)
        self.c2 = np.cosh(self.theta2, dtype = self.dtype)
        """
        """
        column ordering used here: L: 1357 2468, D: 8246 1357, U: 1357 2468, R: 1357 2468
        transfer matrices fail badly under case above here.
        """
        """
        self.transfer_LU1 = np.diag([self.c1]*self.R+[self.c2]*self.L)
        self.transfer_LR1 = np.diag([self.s1]*self.R+[self.s2]*self.L)
        self.transfer_DU1 = np.block([[np.zeros(shape=(self.R,self.R)), np.diag([self.s1]*self.R)],[np.roll(np.diag([self.s2]*self.L),1,axis=1),np.zeros(shape=(self.L,self.L))]])
        self.transfer_DR1 = np.block([[np.zeros(shape=(self.R,self.R)), np.diag([self.c1]*self.R)],[np.roll(np.diag([self.c2]*self.L),1,axis=1),np.zeros(shape=(self.L,self.L))]])
        self.unsolved1 = np.block([[self.transfer_LR1,self.transfer_DR1],[self.transfer_LU1,self.transfer_DU1]])
        print("oh no", np.identity(self.R+self.L)-self.transfer_DU1)
        self.transfer1 = self.transfer_LR1 + self.transfer_DR1@LA.inv(np.identity(self.R+self.L)-self.transfer_DU1)@self.transfer_LU1
        self.transfer_LU2 = np.diag([self.c2]*self.R+[self.c1]*self.L)
        self.transfer_LR2 = np.diag([self.s2]*self.R+[self.s1]*self.L)
        self.transfer_DU2 = np.block([[np.zeros(shape=(self.R,self.R)), np.diag([self.s2]*self.R)],[np.roll(np.diag([self.s1]*self.L),1,axis=1),np.zeros(shape=(self.L,self.L))]])
        self.transfer_DR2 = np.block([[np.zeros(shape=(self.R,self.R)), np.diag([self.c2]*self.R)],[np.roll(np.diag([self.c1]*self.L),1,axis=1),np.zeros(shape=(self.L,self.L))]])
        self.transfer2 = self.transfer_LR2 + self.transfer_DR2@LA.inv(np.identity(self.R+self.L)-self.transfer_DU2)@self.transfer_LU2
        #print(self.transfer1, self.transfer2)
        self.transfer = self.transfer1@self.transfer2
        print(self.unsolved1)
        print(self.transfer1)
        """
    def transfer_blocks(self, N=1):
        for blocks in self.scattering_blocks(N):
            yield chiral_network_layer.blockconversion(blocks)
    def scattering_blocks(self, N=1):
        for i in range(N):
            #print(i)
            yield self.scattering

class magic_square_layer(chiral_network_layer):
    def __init__(self, N, beta=0.0, periodic=True, dtype = np.dtype(np.complex128)):
        assert periodic, "square layer only defined for periodic conditions"
        assert N%2 == 0, "square model only defined for even widths"
        L = N//2
        R = L
        chiral_network_layer.__init__(self, L, R,  dtype = dtype)
        """calculation of important constants. random phase shifts are always outer"""
        theta = beta + np.pi/4
        self.s = np.sin(theta, dtype = self.dtype)
        self.c = np.cos(theta, dtype = self.dtype)
        phi1 = np.ones(R)
        phi2 = np.ones(R)
        phi3 = np.ones(R)
        phi4 = np.ones(R)
        M = np.diag(phi2)
        Y = np.roll(np.diag(phi4),1,axis=0)
        #A13 means layer A in 1 out 3
        self.A11 = (M*self.c**2+Y*self.s**2)*phi1
        self.A33 = (M*self.s**2+Y*self.c**2)*phi3
        self.A13 = (M-Y)*(self.s*self.c*phi3)
        self.A31 = (M-Y)*(self.s*self.c*phi1)
        phi1 = np.ones(R)
        phi2 = np.ones(R)
        phi3 = np.ones(R)
        phi4 = np.ones(R)
        M = np.diag(phi4)
        Y = np.roll(np.diag(phi2),1,axis=1)
        #A13 means layer A in 1 out 3
        self.B11 = (Y*self.c**2+M*self.s**2)*phi1
        self.B33 = (Y*self.s**2+M*self.c**2)*phi3
        self.B13 = (Y-M)*(self.s*self.c*phi3)
        self.B31 = (Y-M)*(self.s*self.c*phi1)
        #check scattering
        scat1 = [[self.A31,self.A33],[self.A11,self.A13]]
        scat2 = [[self.B11,self.B13],[self.B31,self.B33]]
        self.scattering = chiral_network_layer.combine_scattering_matrices(scat1, scat2)
        temp = np.block(self.scattering)
        print("check unitarity", temp.conj().T@temp)
        ## DEBUG:
    def transfer_blocks(self, N=1):
        for blocks in self.scattering_blocks(N):
            yield chiral_network_layer.blockconversion(blocks)
    def scattering_blocks(self, N=1):
        for i in range(N):
            #print(i)
            yield self.scattering

class magic_square_layer_old(chiral_network_layer):
    def __init__(self, N, beta=0.0, periodic=True, dtype = np.dtype(np.complex128)):
        assert periodic, "square layer only defined for periodic conditions"
        assert N%2 == 0, "square model only defined for even widths"
        L = N//2
        R = L
        chiral_network_layer.__init__(self, L, R,  dtype = dtype)
        """calculation of important constants. random phase shifts are always outer"""
        theta = beta + np.pi/4
        self.s = np.sin(theta, dtype = self.dtype)
        self.c = np.cos(theta, dtype = self.dtype)
        Zl1 = np.ones(R)
        Zr1 = np.ones(R)
        Zl2 = np.ones(R)
        Zr2 = np.ones(R)
        cyc1 = np.roll(np.identity(R),1,axis=0)*Zr2
        match1 = np.diag(Zl2)
        self.layer1LL = cyc1*self.c**2-match1*self.s**2
        self.layer1RR = match1*self.c**2-cyc1*self.s**2
        self.layer1RL = (cyc1+match1)*self.s*self.c
        self.layer1LR = -self.layer1RL
        scat1 = [[self.layer1LL*Zl1,self.layer1RL*Zr1],[self.layer1LR*Zl1,self.layer1RR*Zr1]]
        Xl1 = np.ones(R)
        Xr1 = np.ones(R)
        Xl2 = np.ones(R)
        Xr2 = np.ones(R)
        cyc2 = np.roll(np.identity(R),1,axis=0)*Xr2
        match2 = np.diag(Xl2)
        self.layer2LL = cyc2*self.s**2-match2*self.c**2
        self.layer2RR = match2*self.s**2-cyc2*self.c**2
        self.layer2RL = (cyc2+match2)*self.s*self.c
        self.layer2LR = -self.layer2RL
        scat2 = [[self.layer2LL*Zl1,self.layer2RL*Zr1],[self.layer2LR*Zl1,self.layer2RR*Zr1]]
        self.scattering = chiral_network_layer.combine_scattering_matrices(scat1, scat2)
    def transfer_blocks(self, N=1):
        for blocks in self.scattering_blocks(N):
            yield chiral_network_layer.blockconversion(blocks)
    def scattering_blocks(self, N=1):
        for i in range(N):
            #print(i)
            yield self.scattering

class noisy_square_layer_old(chiral_network_layer):
    def __init__(self, N, beta=0.0, periodic=True, dtype = np.dtype(np.complex128), W=0.0, noisetype="chiral"):
        assert periodic, "square layer only defined for periodic conditions"
        assert N%2 == 0, "square model only defined for even widths"
        self.W = W
        self.noisetype = noisetype
        L = N//2
        R = L
        chiral_network_layer.__init__(self, L, R,  dtype = dtype)
        """calculation of important constants. random phase shifts are always outer"""
        theta = beta + np.pi/4
        self.s = np.sin(theta, dtype = self.dtype)
        self.c = np.cos(theta, dtype = self.dtype)
        self.s_q = self.s**2
        self.c_q = self.c**2
        self.sc = self.s*self.c
    def transfer_blocks(self, N=1):
        for blocks in self.scattering_blocks(N):
            yield chiral_network_layer.blockconversion(blocks)
    def scattering_blocks(self, N=1):
        noisetype = self.noisetype
        W = 2*self.W
        R = self.R
        for i in range(N):
            if noisetype=="full_edge":
                Zl1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zr1 = np.ones(R)
                Zl2 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zr2 = np.ones(R)
                Xl1 = np.ones(R)
                Xr1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Xl2 = np.ones(R)
                Xr2 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
            elif noisetype=="chaos":
                Zl1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zr1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zl2 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zr2 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Xl1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Xr1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Xl2 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Xr2 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
            elif noisetype=="single_edge":
                Zl1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zr1 = np.ones(R)
                Zl2 = np.ones(R)
                Zr2 = np.ones(R)
                Xl1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Xr1 = np.ones(R)
                Xl2 = np.ones(R)
                Xr2 = np.ones(R)
            elif noisetype=="AV_rand":
                Zl1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zr1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zl2 = np.ones(R)
                Zr2 = np.ones(R)
                Xl1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Xr1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Xl2 = np.ones(R)
                Xr2 = np.ones(R)
            else:
                Zl1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zr1 = np.ones(R)
                Zl2 = np.ones(R)
                Zr2 = np.ones(R)
                Xl1 = np.ones(R)
                Xr1 = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Xl2 = np.ones(R)
                Xr2 = np.ones(R)
            cyc1 = np.roll(np.diag(Zr2),1,axis=0)
            match1 = np.diag(Zl2)
            self.layer1LL = cyc1*self.c_q-match1*self.s_q
            self.layer1RR = match1*self.c_q-cyc1*self.s_q
            self.layer1RL = (cyc1 + match1)*self.sc
            self.layer1LR = -self.layer1RL
            scat1 = [[self.layer1LR*Zl1,self.layer1RR*Zr1],[self.layer1LL*Zl1,self.layer1RL*Zr1]]
            cyc2 = np.roll(np.diag(Xr2),1,axis=0)
            match2 = np.diag(Xl2)
            self.layer2LL = cyc2*self.s_q-match2*self.c_q
            self.layer2RR = match2*self.s_q-cyc2*self.c_q
            self.layer2RL = (cyc2 + match2)*self.sc
            self.layer2LR = -self.layer2RL
            scat2 = [[self.layer2LR*Zl1,self.layer2RR*Zr1],[self.layer2LL*Zl1,self.layer2RL*Zr1]]
            yield chiral_network_layer.combine_scattering_matrices(scat1, scat2)

class full_network:
    def __init__(self, network_layer):
        self.network_layer = network_layer
    def combine_scattering_matrices(blocks1, blocks2):
        """for converted matrices, returns unitary matrix
        C(U_s & U_o)= C(U_o)@C(U_s)
        where C is the deconversion algorithm
        self => 2
        other => 1
        """
        try:
            a1 = blocks1[0][0]
            b1 = blocks1[0][1]
            c1 = blocks1[1][0]
            d1 = blocks1[1][1]
            a2 = blocks2[0][0]
            b2 = blocks2[0][1]
            c2 = blocks2[1][0]
            d2 = blocks2[1][1]
            m, n = b2.shape
            prefix2 = d2@LA.inv(np.identity(n)-c1@b2)
            prefix1 = a1@LA.inv(np.identity(m)-b2@c1)
        except LA.LinAlgError:
            print(blocks1, blocks2)
        return [[prefix1@a2,b1+prefix1@(b2@d1)],[c2+prefix2@(c1@a2),prefix2@d1]]
    def transfer_eigenvalues(self, samples = 100, presamples = 100):
        size = self.network_layer.R+self.network_layer.L
        Q = np.array(np.random.normal(size=(size, size))+np.random.normal(size=(size, size))*1j, dtype=self.network_layer.dtype)
        scaler = 0
        for component in self.network_layer.transfer(presamples):
            Q, R = np.linalg.qr(component@Q)
            initial = Q
        for component in self.network_layer.transfer(samples):
            Q, R = np.linalg.qr(component@Q)
            scaler += np.log(np.abs(np.diag(R)))
        return scaler/samples, initial
    def tree_scattering(self, length = 32, generator = None):
        """To do: replace lower order computations with direct phase flipping"""
        if generator == None:
            generator = self.network_layer.scattering_blocks(length)
        if length == 1:
            return generator.__next__()
        else:
            return full_network.combine_scattering_matrices(self.tree_scattering(length=length//2, generator=generator),self.tree_scattering(length=length-length//2, generator=generator))
    def conductance(self, length = 20, collection=4, noise_average=1, error=False, original = False):
        if noise_average>1:
            R = []
            L = []
            for i in range(noise_average):
                r, l, trash1, trash2 = self.conductance(length, collection, noise_average=1)
                R.append(r)
                L.append(l)
            result = []
            result.append(np.sum(R)/noise_average)
            result.append(np.sum(L)/noise_average)
            if error:
                result.append(np.std(R, ddof = 1)/np.sqrt(noise_average)+1/10/noise_average)
                result.append(np.std(L, ddof = 1)/np.sqrt(noise_average)+1/10/noise_average)
            if original:
                result.append(R)
                result.append(L)
            return result
        """returns conductance divided by e^2/h"""
        scattering = self.tree_scattering(length=length)
        Z = np.block(scattering)
        #print(Z.conj().T@Z)
        right_trans = np.trace(scattering[0][0]@scattering[0][0].conj().T)
        left_trans = np.trace(scattering[1][1]@scattering[1][1].conj().T)
        return right_trans, left_trans, 0.1, 0.1
    def tracer_conductance(self, order = 5, collection=4, noise_average=1, error=False, original = False, mode = "ansatz"):
        if noise_average>1:
            R = []
            L = []
            for i in range(noise_average):
                r, l, trash1, trash2 = self.tracer_conductance(order, collection, noise_average=1)
                R.append(r)
                L.append(l)
            result = []
            result.append(np.sum(R)/noise_average)
            result.append(np.sum(L)/noise_average)
            if error:
                result.append(np.std(R, ddof = 1)/np.sqrt(noise_average)+1/10/noise_average)
                result.append(np.std(L, ddof = 1)/np.sqrt(noise_average)+1/10/noise_average)
            if original:
                result.append(R)
                result.append(L)
            return result
        """returns conductance divided by e^2/h"""
        scattering = self.tree_scattering(length=int(2**order))
        Z = np.block(scattering)
        #print(Z.conj().T@Z)
        sys_width = len(scattering[0][0])
        vector = np.ones(shape=(sys_width,1))
        if mode == "eigenvalue":
            right_trans = numpy.amax(LA.eig(scattering[0][0].conj().T@scattering[0][0], left=False, right=False))
            left_trans = numpy.amax(LA.eig(scattering[0][0].conj().T@scattering[0][0], left=False, right=False))
        else:
            right_trans = np.trace(vector.T@scattering[0][0].conj().T@scattering[0][0]@vector)/sys_width
            left_trans = np.trace(vector.T@scattering[1][1].conj().T@scattering[1][1]@vector)/sys_width
        return right_trans, left_trans, 0.1, 0.1
    def conductance_eigs(self, order = 5, collection=4):
        """returns conductance divided by e^2/h"""
        scattering = self.tree_scattering(order=order)
        Z = np.block(scattering)
        #print(Z.conj().T@Z)
        right_trans = LA.eig(scattering[0][0]@scattering[0][0].conj().T, left=False, right=False)
        return right_trans
    def detailed_checks(self, order=5):
        scattering = self.tree_scattering(order=order)
        return np.block(scattering)
