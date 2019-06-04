from mpmath import *
import random
import numpy as np
import scipy.linalg as LA
"""scipy linalg inversion seems to be more stable and has better accuracy"""

mp.prec = 4000 #use a with statement to reset precision if needed

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


class network_layer_generator:
    """
    Generates layer matrices of the network. Precomputes as many quantities as possible.
    This base class generates a clean network.
    L is the actual width of the network (multiply by 2 for pairs)
    delta is the t-t' analogue
    norm controls for the magnitude of t, t'

    This currently only generates non-toroidial networks
    """
    def __init__(self, L, delta, norm):
        self.L = L
        self.delta = delta
        self.norm = norm
        self.reset()
    def reset(self):
        sqrt2 = sqrt(2)
        delta2 = self.delta/mpf(2)
        tempa = sqrt(1-2*delta2**2)/sqrt2
        self.theta1 = atanh((tempa-delta2)*self.norm)
        self.theta2 = atanh((tempa+delta2)*self.norm)
        #print("check = 1", tanh(self.theta1)**2+tanh(self.theta2)**2)
        #print("check = delta", tanh(self.theta2)-tanh(self.theta1))
        self.s1 = sinh(self.theta1)
        self.s2 = sinh(self.theta2)
        self.c1 = cosh(self.theta1)
        self.c2 = cosh(self.theta2)
        self.layer_comp()
        self.clean = self.layer1*self.layer2
        if self.layer_comp_t(): self.clean_t = self.layer1_t*self.layer2_t
    """This gives the non-looped version"""
    def layer_comp(self):
        """2n, 2n+1 crossings"""
        layer1 = eye(self.L)
        for segnum in range(self.L//2):
            layer1[segnum*2,segnum*2]=self.c1
            layer1[segnum*2+1,segnum*2+1]=self.c1
            layer1[segnum*2,segnum*2+1]=self.s1
            layer1[segnum*2+1,segnum*2]=self.s1
        self.layer1 = layer1
        layer2 = eye(self.L)
        for segnum in range((self.L-1)//2):
            layer2[segnum*2+1,segnum*2+1]=self.c2
            layer2[segnum*2+2,segnum*2+2]=self.c2
            layer2[segnum*2+2,segnum*2+1]=self.s2
            layer2[segnum*2+1,segnum*2+2]=self.s2
        self.layer2 = layer2
    def layer_comp_t(self):
        if self.L%2!=0:
            print("torus not possible")
            return False
        """2n, 2n+1 crossings"""
        layer1 = eye(self.L)
        for segnum in range(self.L//2):
            layer1[segnum*2,segnum*2]=self.c1
            layer1[segnum*2+1,segnum*2+1]=self.c1
            layer1[segnum*2,segnum*2+1]=self.s1
            layer1[segnum*2+1,segnum*2]=self.s1
        self.layer1_t = layer1
        layer2 = eye(self.L)
        for segnum in range(self.L//2):
            layer2[segnum*2+1,segnum*2+1]=self.c2
            layer2[(segnum*2+2)%self.L,(segnum*2+2)%self.L]=self.c2
            layer2[(segnum*2+2)%self.L,segnum*2+1]=self.s2
            layer2[segnum*2+1,(segnum*2+2)%self.L]=self.s2
        self.layer2_t = layer2
        return True
    def clean_generator(self):
        while True:
            yield self.clean
    def basic_noisy_generator(self, W=0.1):
        """random might be slow; can replace with better idea"""
        trueW = float(sqrt((0.5-W)/2)+1/2)
        while True:
            Z1 = diag([-1 if i%2==0 and random.random()<W else 1 for i in range(self.L)])
            Z2 = diag([-1 if i%2==1 and random.random()<W else 1 for i in range(self.L)])
            yield Z1*self.layer1*Z1*Z2*self.layer2*Z2
    def clean_generator_t(self):
        while True:
            yield self.clean_t
    def basic_noisy_generator_t(self, W=0.1):
        """random might be slow; can replace with better idea"""
        trueW = float(sqrt((0.5-W)/2)+1/2)
        while True:
            Z1 = diag([-1 if i%2==0 and random.random()<W else 1 for i in range(self.L)])
            Z2 = diag([-1 if i%2==1 and random.random()<W else 1 for i in range(self.L)])
            yield Z1*self.layer1_t*Z1*Z2*self.layer2_t*Z2
class network_layer_generator_np:
    """
    Generates layer matrices of the network. Precomputes as many quantities as possible.
    This base class generates a clean network.
    L is the actual width of the network (multiply by 2 for pairs)
    delta is the t-t' analogue
    norm controls for the magnitude of t, t'

    This currently only generates non-toroidial networks
    """
    def __init__(self, L, delta, norm, dtype=np.dtype(np.complex128)):
        self.dtype = dtype
        self.L = L
        self.delta = delta
        self.norm = norm
        self.reset()
    def reset(self):
        sqrt2 = np.sqrt(2, dtype = self.dtype)
        delta2 = self.delta/2
        tempa = np.sqrt(1-2*delta2**2, dtype = self.dtype)/sqrt2
        self.theta1 = np.arctanh((tempa-delta2)*self.norm, dtype = self.dtype)
        self.theta2 = np.arctanh((tempa+delta2)*self.norm, dtype = self.dtype)
        #print("check = 1", tanh(self.theta1)**2+tanh(self.theta2)**2)
        #print("check = delta", tanh(self.theta2)-tanh(self.theta1))
        self.s1 = np.sinh(self.theta1)
        self.s2 = np.sinh(self.theta2)
        self.c1 = np.cosh(self.theta1)
        self.c2 = np.cosh(self.theta2)
        self.layer_comp()
        self.clean = self.layer1.dot(self.layer2)
        if self.layer_comp_t(): self.clean_t = self.layer1_t.dot(self.layer2_t)
    """This gives the non-looped version"""
    def layer_comp(self):
        """2n, 2n+1 crossings"""
        layer1 = np.identity(self.L, dtype = self.dtype)
        for segnum in range(self.L//2):
            layer1[segnum*2,segnum*2]=self.c1
            layer1[segnum*2+1,segnum*2+1]=self.c1
            layer1[segnum*2,segnum*2+1]=self.s1
            layer1[segnum*2+1,segnum*2]=self.s1
        self.layer1 = layer1
        layer2 = np.identity(self.L, dtype = self.dtype)
        for segnum in range((self.L-1)//2):
            layer2[segnum*2+1,segnum*2+1]=self.c2
            layer2[segnum*2+2,segnum*2+2]=self.c2
            layer2[segnum*2+2,segnum*2+1]=self.s2
            layer2[segnum*2+1,segnum*2+2]=self.s2
        self.layer2 = layer2
    def layer_comp_t(self):
        if self.L%2!=0:
            print("torus not possible")
            return False
        """2n, 2n+1 crossings"""
        layer1 = np.identity(self.L, dtype = self.dtype)
        for segnum in range(self.L//2):
            layer1[segnum*2,segnum*2]=self.c1
            layer1[segnum*2+1,segnum*2+1]=self.c1
            layer1[segnum*2,segnum*2+1]=self.s1
            layer1[segnum*2+1,segnum*2]=self.s1
        self.layer1_t = layer1
        layer2 = np.identity(self.L, dtype = self.dtype)
        for segnum in range(self.L//2):
            layer2[segnum*2+1,segnum*2+1]=self.c2
            layer2[(segnum*2+2)%self.L,(segnum*2+2)%self.L]=self.c2
            layer2[(segnum*2+2)%self.L,segnum*2+1]=self.s2
            layer2[segnum*2+1,(segnum*2+2)%self.L]=self.s2
        self.layer2_t = layer2
        return True
    def clean_generator(self):
        while True:
            yield self.clean
    def basic_noisy_generator(self, W=0.1):
        """random might be slow; can replace with better idea"""
        while True:
            Z1 = np.diag([-1 if i%2==0 and random.random()<W else 1 for i in range(self.L)])
            Z2 = np.diag([-1 if i%2==1 and random.random()<W else 1 for i in range(self.L)])
            yield (Z1.dot(self.layer1).dot(Z1)).dot(Z2.dot(self.layer2).dot(Z2))
    def chiral_flip_noisy_generator(self, W=0.1):
        """random might be slow; can replace with better idea"""
        while True:
            Z1 = np.diag([-1 if i%2==0 and random.random()<W else 1 for i in range(self.L)])
            Z2 = np.diag([-1 if i%2==0 and random.random()<W else 1 for i in range(self.L)])
            yield (self.layer1.dot(Z1)).dot(self.layer2.dot(Z2))
    def chiral_cyc_noisy_generator(self, W=0.1):
        """random might be slow; can replace with better idea"""
        trueW = 2*W
        while True:
            Z1 = np.diag([np.exp(1j*(np.random.random()-0.5)*trueW) if i%2==0 else 1 for i in range(self.L)])
            Z2 = np.diag([np.exp(1j*(np.random.random()-0.5)*trueW) if i%2==0 else 1 for i in range(self.L)])
            yield (self.layer1.dot(Z1)).dot(self.layer2.dot(Z2))
    def clean_generator_t(self):
        while True:
            yield self.clean_t
    def basic_noisy_generator_t(self, W=0.1):
        """random might be slow; can replace with better idea"""
        while True:
            Z1 = np.diag([-1 if i%2==0 and random.random()<W else 1 for i in range(self.L)])
            Z2 = np.diag([-1 if i%2==1 and random.random()<W else 1 for i in range(self.L)])
            yield (Z1.dot(self.layer1_t).dot(Z1)).dot(Z2.dot(self.layer2_t).dot(Z2))
    def chiral_flip_noisy_generator_t(self, W=0.1):
        """random might be slow; can replace with better idea"""
        while True:
            Z1 = np.diag([-1 if i%2==0 and random.random()<W else 1 for i in range(self.L)])
            Z2 = np.diag([-1 if i%2==0 and random.random()<W else 1 for i in range(self.L)])
            yield (self.layer1_t.dot(Z1)).dot(self.layer2_t.dot(Z2))
    def chiral_cyc_noisy_generator_t(self, W=0.1):
        """random might be slow; can replace with better idea"""
        trueW = 2*W
        while True:
            Z1 = np.diag([np.exp(1j*(np.random.random()-0.5)*trueW) if i%2==0 else 1 for i in range(self.L)])
            Z2 = np.diag([np.exp(1j*(np.random.random()-0.5)*trueW) if i%2==0 else 1 for i in range(self.L)])
            yield (self.layer1_t.dot(Z1)).dot(self.layer2_t.dot(Z2))
"""computes a long matrix product of length 2**n"""
def long_matrix_product(generator, order=0, index = 0):
    if order==0:
        print("completed: ", index)
        return generator.__next__()
    else:
        #print(order)
        return long_matrix_product(generator, order=order-1, index=index)*long_matrix_product(generator, order=order-1, index=index+2**(order-1))
"""To do: parallelize and use hard disk space"""
