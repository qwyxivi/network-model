import random
import numpy as np
import scipy.linalg as LA
"""scipy linalg inversion seems to be more stable and has better accuracy"""

class pu_block:
    def __init__(self, a, b, c, d, converted=False, c_count = 0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.converted = converted
        self.c_count = c_count
    def matx(self):
        return np.block([[self.a,self.b],[self.c,self.d]])
    def convert(self, flag=True):
        assert not self.converted
        """converts to ordinary signature"""
        a, b, c, d = self.a, self.b, self.c, self.d
        dinv = LA.inv(d)
        #print(LA.eig(d, left=False, right=False))
        bdi = b@dinv
        self.a = a - bdi@c
        self.b = bdi
        self.c = -dinv@c
        self.d = dinv
        self.converted = True
        self.c_count += 1
    def deconvert(self, flag=True):
        assert self.converted
        a, b, c, d = self.a, self.b, self.c, self.d
        dinv = LA.inv(d)
        #print(LA.eig(d, left=False, right=False))
        bdi = b@dinv
        self.a = a - bdi@c
        self.b = bdi
        self.c = -dinv@c
        self.d = dinv
        self.converted = False
        self.c_count += 1
    def combine_c(self, other):
        """for converted matrices, returns unitary matrix
        C(U_s & U_o)= C(U_o)@C(U_s)
        where C is the deconversion algorithm
        """
        assert self.converted
        assert other.converted
        m, n = self.b.shape
        prefix2 = self.d@LA.inv(np.identity(n)-other.c@self.b)
        prefix1 = other.a@LA.inv(np.identity(m)-self.b@other.c)
        return pu_block(prefix1@self.a,other.b+prefix1@(self.b@other.d),self.c+prefix2@(other.c@self.a),prefix2@other.d, converted=True)
    def multiply_c(self, other):
        """for deconverted matrices"""
        assert not self.converted
        assert not other.converted
        return pu_block(other.a@self.a+other.b@self.c,other.a@self.b+other.b@self.d,other.c@self.a+other.d@self.c,other.c@self.b+other.d@self.d)





class chiral_network_layer:
    def blockconversion(blocks):
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
    def scattering_blocks(self, N=1):
        """yields the scattering matrix of a single layer, in blocks"""
        for blocks in self.transfer_blocks(N):
            yield chiral_network_layer.blockconversion(blocks)
    def transfer(self, N=1):
        """yields the transfer matrix of a single layer"""
        for blocks in self.transfer_blocks(N):
            yield np.block(blocks)
    def scattering(self, N=1):
        for blocks in self.scattering_blocks(N):
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
            print(i)
            yield [[self.rr, self.rl],[self.lr, self.ll]]
    def scattering_blocks(self, N=1):
        scat = chiral_network_layer.blockconversion([[self.rr, self.rl],[self.lr, self.ll]])
        for i in range(N):
            print(i)
            yield scat
    def transfer(self, N=1):
        """yields the transfer matrix of a single layer"""
        result = np.block([[self.rr, self.rl],[self.lr, self.ll]])
        for i in range(N):
            print(i)
            yield result
    def scattering(self, N=1):
        result = np.block(chiral_network_layer.blockconversion([[self.rr, self.rl],[self.lr, self.ll]]))
        for i in range(N):
            print(i)
            yield result

class noisy_flat_layer(clean_flat_layer):
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
        elif self.noisetype == "random_iso_flip":
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
        elif self.noisetype == "random_chiral_flip":
            for i in range(N):
                Z1r = np.array([-1 if np.random.random()<W else 1 for i in range(self.R)])
                Z2r = np.array([-1 if np.random.random()<W else 1 for i in range(self.R)])
                rr = (self.rr1*Z1r)@self.rr2*Z2r+self.rl1@self.lr2*Z2r
                rl = self.rl1@self.ll2+(self.rr1*Z1r)@self.rl2
                lr = (self.lr1*Z1r)@self.rr2*Z2r+self.ll1@self.lr2*Z2r
                ll = self.ll1@self.ll2+(self.lr1*Z1r)@self.rl2
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
        elif self.noisetype == "random_iso_rotation":
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
        elif self.noisetype == "random_chiral_rotation":
            for i in range(N):
                Z1r = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                Z2r = np.array([np.exp(1j*(np.random.random()-0.5)*W) for i in range(self.R)])
                rr = (self.rr1*Z1r)@self.rr2*Z2r+self.rl1@self.lr2*Z2r
                rl = self.rl1@self.ll2+(self.rr1*Z1r)@self.rl2
                lr = (self.lr1*Z1r)@self.rr2*Z2r+self.ll1@self.lr2*Z2r
                ll = self.ll1@self.ll2+(self.lr1*Z1r)@self.rl2
                yield [[rr, rl],[lr, ll]]

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
        if generator == None:
            generator = self.network_layer.scattering_blocks(2**order)
        if order == 0:
            return generator.__next__()
        else:
            return full_network.combine_scattering_matrices(self.tree_scattering(order=order-1, generator=generator),self.tree_scattering(order=order-1, generator=generator))
