# Python code to reproduce experiments in the paper
import numpy as np
import funcs
import os

CODEPATH = 'PTMcode'
trainingfile = '../Reuters/train-data.dat' #training data set
testfile = '../Reuters/test-data.dat' #test data set

Mmax = 100 # Max num of topics
Mmin = 10 # Min num of topics
step = 5 # Topic reduction step size
seed0 = 1819114101
np.random.seed(seed0)

refile = open('results.dat', 'w+')
refile.write('M, bic, Etr_lkh, ho_lkh, avg_tpc/doc, avg_wrd/tpc, num_unq_wrds, runtime, ')
refile.write('uv_avg_tpc/doc, uv_avg_wrd/tpc, uv_num_unq_wrds\n')
refile.close()


#main loop

k = reversed(range(Mmin,Mmax+step,step))
for M in k:
    path = 'dir' + str(M)
    ## run PTM on training data
    seed = np.random.randint(seed0)
    cmdtxt = CODEPATH + '/ptm --num_topics ' + str(M) + ' --corpus ' + trainingfile + ' --convergence 3e-4 --seed '+str(seed)
    if M == Mmax:
        cmdtxt = cmdtxt + ' --init seeded --dir ' + path
    else:
        cmdtxt = cmdtxt + ' --init load --model ' + path + '/init --dir ' + path 
    #if M <Mmax:# 25:
        #seed = np.random.randint(seed0)
        #continue
    os.system(cmdtxt)

    ### read training topic proportions
    theta = np.loadtxt(path+'/final.alpha')
    vswitch = np.loadtxt(path+'/final.v') 
    #read word probabilities
    beta = np.exp(np.loadtxt(path+'/final.beta'))
    uswitch = np.loadtxt(path+'/final.u') 
    for j in range(M):
        ind = np.where(uswitch[:,j]==0)[0]
        beta[ind,j+1] = beta[ind,0]
    N = beta.shape[0]
   
    
    # compute sparsity measures
    (avg_tpcs, avg_wrds, unq_wrds) = funcs.topic_word_sparsity(path+'/word-assignments.dat',N,M,uswitch)
    (uv_avg_tpcs, uv_avg_wrds, uv_unq_wrds) = funcs.switch_topic_word_sparsity(uswitch,vswitch,N,M)
    
    # read training likelihood
    lk = np.loadtxt(path+'/likelihood.dat')
    bic = lk[-1,0]
    lkh = lk[-1,1]
    runtime = lk[-1,3]

    # inference on test set
    seed = np.random.randint(seed0)
    cmdtxt = CODEPATH + '/ptm --task test' + ' --corpus ' + testfile + ' --convergence 1e-4 --seed '+str(seed)
    cmdtxt = cmdtxt + ' --dir ' + path + ' --model ' + path + '/final' 
    os.system(cmdtxt) 
      
    ## compute likelihood on training set
    Etrlk = funcs.compute_lkh(trainingfile, beta[:,1:M+1], theta)
   
    
    ## save useful stuff
    # results file
    refile = open('results.dat', 'a')
    refile.write(str(M) + ', ' + str(bic) + ', ' + str(Etrlk) + ', ' + str(avg_tpcs) + ', ')
    refile.write(str(np.mean(avg_wrds)) + ', ' + str(np.sum(unq_wrds)) + ', ')
    refile.write(str(runtime)+', '+str(uv_avg_tpcs) + ', ' + str(np.mean(uv_avg_wrds)) + ', ' + str(np.sum(uv_unq_wrds)) + '\n')
    refile.close()
    
    ## prepare for the next model order (writes it in the next folder)
    if  (M >= (Mmin+step)):
        next_path = 'dir' + str(M-step)
        os.system('mkdir -p '+next_path)
        funcs.prepare_next_forptm(step, path, next_path, theta)

    ## delete other files
    #os.system('rm -rf '+ path)
    
res = np.loadtxt('results.dat',skiprows=1,delimiter=',')
am = np.argmin(res[:,1])
print('M* = %d' % int(res[am,0]))
    
