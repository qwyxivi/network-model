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

import numpy as np

def logging(description,data):
    with open("chaineig.log","a+") as file:
        file.write(description)
        file.write(str(data))
        file.write("\n")
"""for mpmath matrices, complex vectors"""
"""TO DO: convergence tracking"""
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
