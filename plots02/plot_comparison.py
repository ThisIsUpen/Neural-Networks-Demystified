import matplotlib.pyplot as plt
import pylab

methods = ('SGD', 'ASGD', 'LBFGS', 'NAG')

#collect data from each log file
dl = [1]
for name in methods:
    filename = name+'.log'
    data = pylab.loadtxt(filename)
    dl[0] = data[1:,0]
    dl.append(data[1:,1])

#plot data
for i in range(1,len(dl)):
    pylab.plot(dl[0], dl[i])

pylab.xlabel('iterations');
pylab.ylabel('cost function');
pylab.legend([ methods[0], methods[1], methods[2], methods[3] ])
pylab.show()

