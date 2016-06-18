import os
import numpy as np
 
# download and compile lda code
# from http://www.cs.columbia.edu/~blei/lda-c/lda-c-dist.tgz


code = 'lda-c-dist'

M = 50 # number of topics

trainfile = 'Reuters/train-data.dat'
testfile = 'Reuters/test-data.dat'

# write settings file
fp = open('%s/settings.txt' %code,'w')
fp.write('var max iter 20\nvar convergence 1e-4\nem max iter 100\nem convergence 1e-4\nalpha estimate')
fp.close()

# train lda
cmdtxt = '%s/lda est 1.0 %d %s/settings.txt %s seeded dir' %(code,M,code,trainfile)
os.system(cmdtxt)

# read topic proportions and save train-lda-data.dat
train_gamma = np.loadtxt('dir/final.gamma')
train_gamma /= np.sum(train_gamma,1).reshape(-1,1)
np.savetxt('Reuters/train-lda-data.dat',train_gamma, '%f')


# save test-lda-data.dat
cmdtxt = '%s/lda inf %s/settings.txt dir/final %s dir/test' %(code,code,testfile)
os.system(cmdtxt)

test_gamma = np.loadtxt('dir/test-gamma.dat')
test_gamma /= np.sum(test_gamma,1).reshape(-1,1)
np.savetxt('Reuters/test-lda-data.dat',test_gamma, '%f')


os.system('rm -r dir')
