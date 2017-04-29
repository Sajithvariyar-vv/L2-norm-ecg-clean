
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import pinv
from scipy.sparse import *
from scipy import *

#loading data from the file
signal = np.loadtxt('data/ecg2.txt')
#getting the length

N=len(signal)

# Smoothing (degree = 2)
# D is the second-order difference matrix.
# It approximates the second-order derivative.
# In order to exploit fast banded solvers in Matlab,
# we define D as a sparse matrix using 'spdiags'.
d=diags([1, -2, 1], [0, 1, 2], shape=(N-2,N)).todense()
d_trans=d.transpose()
lam = 50
eye=diags([1], 0,shape=(N,N)).todense()
F = eye+ lam * d_trans * d

#x = pinv(F)*signal.transpose()
w=pinv(F)
x=w.dot(signal)
q=x.T
#print eye
#print q.shape
#print x
#print w.shape
#print signal.shape 
#print F.shape
#print len(signal)
#print signal.shape

plt.figure(1)
plt.subplot(211)
plt.plot(signal,'b')
plt.title('Orginal ECG Signal')
plt.ylabel('ECG Samples')
plt.xlabel('Sampling rate')

plt.subplot(313)
plt.plot(q, 'r')
plt.title('Denoised ECG Signal')
plt.ylabel('ECG Samples')
plt.xlabel('Sampling rate')
plt.show()

