import random
import numpy as np
import scipy.linalg as LA
"""scipy linalg inversion seems to be more stable and has better accuracy"""
import collections

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
        self.theta2 = np.arctanh((tempa+delta2)*self.norm, dtype = self.dtype)
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

class clean_uniform_layer(chiral_network_layer):
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
        self.theta2 = np.arctanh((tempa+delta2)*self.norm, dtype = self.dtype)
        self.RLlinks = np.zeros(shape=(self.L, self.R))
        for i in range(min(self.L, self.R)):
            self.RLlinks[i,i]=self.theta1
        if not self.periodic:
            for i in range(min(self.L, self.R-1)):
                self.RLlinks[i+1,i]=self.theta2
        else:
            for i in range(self.L):
                self.RLlinks[(i+1)%self.L,i]=self.theta2
        self.transfer = LA.expm(np.block([[np.zeros(shape=(self.R,self.R)),self.RLlinks],[self.RLlinks.T,np.zeros(shape=(self.L,self.L))]]))
        self.rr = self.transfer[:self.R,:self.R]
        self.rl = self.transfer[:self.R,self.R:]
        self.lr = self.transfer[self.R:,:self.R]
        self.ll = self.transfer[self.R:,self.R:]
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
        result = self.transfer
        for i in range(N):
            #print(i)
            yield result
    def scattering(self, N=1):
        result = np.block(chiral_network_layer.blockconversion([[self.rr, self.rl],[self.lr, self.ll]]))
        for i in range(N):
            #print(i)
            yield result

class noisy_uniform_layer(chiral_network_layer):
    def __init__(self, N, delta = 0.0, norm = 1.0, periodic=False, dtype = np.dtype(np.complex128), W=0, noisetype="paired_flip"):
        self.W = W
        self.noisetype = noisetype
        clean_uniform_layer.__init__(self, N, delta, norm, periodic, dtype)
    def transfer_blocks(self, N=1):
        W = self.W
        if self.noisetype == "iso_flip":
            for i in range(N):
                Zr = np.array([-1 if np.random.random()<W else 1 for i in range(self.R)])
                Zl = np.array([-1 if np.random.random()<W else 1 for i in range(self.L)])
                rr = self.rr*Zr
                rl = self.rl*Zl
                lr = self.lr*Zr
                ll = self.ll*Zl
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "chiral_flip":
            for i in range(N):
                Zr = np.array([-1 if np.random.random()<W else 1 for i in range(self.R)])
                rr = self.rr*Zr
                rl = self.rl
                lr = self.lr*Zr
                ll = self.ll
                yield [[rr, rl],[lr, ll]]
        W = 2*self.W
        if self.noisetype == "iso_rotation":
            for i in range(N):
                Zr = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Zl = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.L)])
                rr = self.rr*Zr
                rl = self.rl*Zl
                lr = self.lr*Zr
                ll = self.ll*Zl
                yield [[rr, rl],[lr, ll]]
        elif self.noisetype == "chiral_rotation":
            for i in range(N):
                Zr = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                rr = self.rr*Zr
                rl = self.rl
                lr = self.lr*Zr
                ll = self.ll
                yield [[rr, rl],[lr, ll]]

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
    def tree_scattering(self, order = 5, generator = None):
        """To do: replace lower order computations with direct phase flipping"""
        if generator == None:
            generator = self.network_layer.scattering_blocks(2**order)
        if order == 0:
            return generator.__next__()
        else:
            return full_network.combine_scattering_matrices(self.tree_scattering(order=order-1, generator=generator),self.tree_scattering(order=order-1, generator=generator))
    def conductance(self, order = 5, collection=4, noise_average=1, error=False, original = False):
        if noise_average>1:
            R = []
            L = []
            for i in range(noise_average):
                r, l = self.conductance(order, collection, noise_average=1)
                R.append(r)
                L.append(l)
            result = []
            result.append(np.sum(R)/noise_average)
            result.append(np.sum(L)/noise_average)
            if error:
                result.append(np.std(R, ddof = 1))
                result.append(np.std(L, ddof = 1))
            if original:
                result.append(R)
                result.append(L)
            return result
        """returns conductance divided by e^2/h"""
        scattering = self.tree_scattering(order=order)
        Z = np.block(scattering)
        #print(Z.conj().T@Z)
        right_trans = np.trace(scattering[0][0]@scattering[0][0].conj().T)
        left_trans = np.trace(scattering[1][1]@scattering[1][1].conj().T)
        return right_trans, left_trans
    def conductance_eigs(self, order = 5, collection=4):
        """returns conductance divided by e^2/h"""
        scattering = self.tree_scattering(order=order)
        Z = np.block(scattering)
        #print(Z.conj().T@Z)
        right_trans = LA.eig(scattering[0][0]@scattering[0][0].conj().T, left=False, right=False)
        return right_trans
