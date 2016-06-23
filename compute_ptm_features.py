import numpy as np
import re, sys, os


M = 50
path = 'PTM/dir%d' %M
#read word probabilities
theta = np.loadtxt(path+'/final.alpha')
vswitch = np.loadtxt(path+'/final.v') 
beta = np.exp(np.loadtxt(path+'/final.beta'))
uswitch = np.loadtxt(path+'/final.u') 
for j in range(M):
	ind = np.where(uswitch[:,j]==0)[0]
	beta[ind,j+1] = beta[ind,0]
beta /= np.sum(beta,0)

'''Ntotal = 500 #reduced vocabulary size
# take x topic-specific words from each topic; 
# choose x proprotional to the frequency with which that topic is used in training
NperTopic = Ntotal*np.sum(theta,0)/np.sum(theta)

selected_words = set()
for j in range(M):
	ind = np.where(uswitch[:,j]==1)[0]
	temp = ind[np.argsort(-beta[ind,j+1])]
	for x in temp[:NperTopic[j]]:
		selected_words.add(x)
'''
# Take all words which are topic-specific in at least one topic
selected_ind = np.where(np.any(uswitch==1,1)==True)[0]

# read old vocabs
old_vocabs = [x.split(',')[0] for x in open('Reuters/vocabs.txt').readlines()]

# write new words:
fp = open('Reuters/ptm_vocabs.txt','w')
fp2 = open('Reuters/ptm_vocabs_stats.txt','w')
wrd_map = {}
for n,w in enumerate(selected_ind):
	wrd_map.update({w:n})
	temp = np.fabs(beta[w,np.where(uswitch[w,:]==1)[0]+1]-beta[w,0])
	fp2.write('%.5f %.5f %.5f\n' %(np.max(temp),np.mean(temp),np.std(temp)))
	fp.write('%s, %d' %(old_vocabs[w],n))
fp.close()
fp2.close()

# rewrite documents based on the new dictionary

for name in ['train', 'test', 'valid']:

	fp_in = open('Reuters/%s-data.dat' %name)
	fp_out = open('Reuters/ptm_%s-data.dat' %name,'w')
	fplbl_in = open('Reuters/%s-label.dat' %name)
	fplbl_out = open('Reuters/ptm_%s-label.dat' %name,'w')
	while True:
		doc = fp_in.readline()
		if len(doc) == 0:
			break
		lbls = fplbl_in.readline()
		wrds = re.findall('([0-9]*):[0-9]*',doc) 
		cnts = re.findall('[0-9]*:([0-9]*)',doc) 
		doc_new = {}
		for n,w in enumerate(wrds):
			try:
				w_new = wrd_map[int(w)]
				doc_new.update({w_new:cnts[n]})
			except KeyError:
				continue
		ld = len(doc_new)
		if ld > 0:
			doc_txt = str(ld) + ' ' + ' '.join(['%d:%s' %(w,doc_new[w]) for w in doc_new.keys()])
		else:
			continue	

		fp_out.write(doc_txt+'\n')
		fplbl_out.write(lbls)
	fp_in.close()
	fp_out.close()
	fplbl_in.close()
	fplbl_out.close()


