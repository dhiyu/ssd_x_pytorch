import _pickle as cPickle
import matplotlib.pyplot as plt
import numpy as np

filepath = '../ssd300_120000/test/'
plt.figure()
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR cruve')
classname = ['Iron lighter', 'Black nail lighter', 'Cutter', 'Power and battery', 'Scissor']
for i in range(0, 5):
    fr = open(filepath + str(i+1) + '_pr.pkl', 'rb')
    inf = cPickle.load(fr)
    fr.close()

    x = inf['rec']
    y = inf['prec']
    print(type(x))
    plt.plot(x, y, label=classname[i])
    print('AP：', inf['ap'])

x = np.linspace(0, 1, 50)
y = x
plt.plot(x, y)
plt.grid()
plt.legend()
plt.show()
plt.close()