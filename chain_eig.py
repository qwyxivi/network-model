"""
correlation length
power law correlation exponents
critical conductance

Limits to check:


Check review papers for localization and exponents to calculate from it.

Try one sided disorder, transition through delta, try varying values for disorder strength. n=3, n=4 work, maybe

Calculate conductivity, search for paper on conductivity from

Read bennecker paper carefully for conductivity
"""


from mpmath import *
import numpy as np

def logging(description,data):
    with open("chaineig.log","a+") as file:
        file.write(description)
        file.write(str(data))
        file.write("\n")

"""for mpmath matrices, complex vectors"""

"""TO DO: convergence tracking"""
"""TO DO: Reduce to single QR"""
def eiglog_mp(matrixgen, size, level = 1, samples=1000, presamples=100):
    if level == 1:
        initial = randmatrix(size, level)-0.5+(randmatrix(size, level)-0.5)*1j
        scaler = 0
        for i in range(presamples):
            #print(matrixgen.__next__()*initial)
            initial = matrixgen.__next__()*initial
            norm = fabs((initial.T*initial)[0,0])
            initial = initial/sqrt(norm)
        for i in range(samples):
            #print(matrixgen.__next__()*initial)
            initial = matrixgen.__next__()*initial
            norm = fabs((initial.T*initial)[0,0])
            scaler += ln(norm)
            initial = initial/sqrt(norm)
        logging("eiglog results: total log:", scaler)
        logging("eiglog results: sample length:", samples)
        logging("eiglog results: final matrix", initial)
        return scaler/samples/2, initial
    initial = randmatrix(size, level)-0.5+(randmatrix(size, level)-0.5)*1j
    scaler = 0
    for i in range(presamples):
        #print(matrixgen.__next__()*initial)
        Q, R = qr(matrixgen.__next__()*initial)
        initial = Q[:,0:level]
        #"""the following can possibly be sped up"""
        #scale += sum([ln(R[i,i]) for i in range(size)])
    for i in range(samples):
        Q, R = qr(matrixgen.__next__()*initial)
        initial = Q[:,0:level]
        """the following can possibly be sped up"""
        scaler += sum([ln(fabs(R[i,i])) for i in range(level)])
    logging("eiglog results: total log:", scaler)
    logging("eiglog results: sample length:", samples)
    logging("eiglog results: final matrix", initial)
    return scaler/samples, initial
def eiglog_np(matrixgen, size, level = 1, samples=1000, presamples=100, dtype=np.dtype(np.complex128)):
    initial = np.array(np.random.normal(size=(size, level))+np.random.normal(size=(size, level))*1j, dtype=dtype)
    scaler = 0
    for i in range(presamples):
        #print(matrixgen.__next__()*initial)
        Q, R = np.linalg.qr(matrixgen.__next__().dot(initial))
        initial = Q[:,0:level]
        #"""the following can possibly be sped up"""
        #scale += sum([ln(R[i,i]) for i in range(size)])
    for i in range(samples):
        Q, R = np.linalg.qr(matrixgen.__next__().dot(initial))
        initial = Q[:,0:level]
        """the following can possibly be sped up"""
        scaler += sum([np.log(np.abs(R[i,i])) for i in range(level)])
    logging("eiglog results: total log:", scaler)
    logging("eiglog results: sample length:", samples)
    logging("eiglog results: final matrix", initial)
    return scaler/samples, initial
def extraction(matrixgen, size, samples=1000, presamples=100):
    a = []
    prev = 0
    for i in range(size):
        values, trash = eiglog_mp(matrixgen, size, i+1, samples, presamples)
        a.append(values-prev)
        prev = values
        if i==size-1:
            print("check == 1", values)
    return a
def extraction_np(matrixgen, size, samples=1000, presamples=100):
    a = []
    prev = 0
    for i in range(size):
        values, trash = eiglog_np(matrixgen, size, i+1, samples, presamples)
        a.append(values-prev)
        prev = values
        if i==size-1:
            print("check == 0", values)
    return a

def extraction_np_fast(matrixgen, size, samples=1000, presamples=100, dtype=np.dtype(np.complex128)):
    level = size
    initial = np.array(np.random.normal(size=(size, level))+np.random.normal(size=(size, level))*1j, dtype=dtype)
    scaler = 0
    for i in range(presamples):
        #print(matrixgen.__next__()*initial)
        Q, R = np.linalg.qr(matrixgen.__next__().dot(initial))
        initial = Q[:,0:level]
        #"""the following can possibly be sped up"""
        #scale += sum([ln(R[i,i]) for i in range(size)])
    for i in range(samples):
        Q, R = np.linalg.qr(matrixgen.__next__().dot(initial))
        initial = Q[:,0:level]
        """the following can possibly be sped up"""
        scaler += np.log(np.abs(np.diag(R)))
    logging("eiglog results: total log:", scaler)
    logging("eiglog results: sample length:", samples)
    logging("eiglog results: final matrix", initial)
    print(scaler)
    return scaler/samples, initial

def conductance_est(matrixgen, size, samples=1000, presamples=100, dtype=np.dtype(np.complex128), negative = False, scale = 1):
    logeigenvalues, basis = extraction_np_fast(matrixgen, size, samples, presamples, dtype)
    cut = (size)//2
    if negative: cut = cut+1
    print(cut)
    parts = logeigenvalues[cut:]
    conductance = 0
    for i in parts:
        print(i)
        conductance += np.exp(2*i*scale)
    return conductance
