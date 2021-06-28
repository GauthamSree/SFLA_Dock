import sys
import argparse

sys.path.insert(0, './support')
parser = argparse.ArgumentParser(description='Molecular conformer generator')
parser.add_argument('-f', required=True, help='sdf output file')
args = parser.parse_args()
i=0
irmsd=20
high_irmsd=9
medium_irmsd=9
accept_irmsd=9
counthigh=0
countmedium=0
countaccept=0
f=open(args.f, 'r')
for line in f:
	
	l=line.split()
	if irmsd>float(l[2]):
		minp=i
		fnat=float(l[1])
		irmsd=float(l[2])
		lrmsd=float(l[3])
		qlty=l[4]
	'''
	#print l
	if l[4]=='High':
		if high_irmsd> float(l[2]):
			minhigh=i
			high_fnat=float(l[1])
			high_irmsd=float(l[2])
			high_lrmsd=float(l[3])
		counthigh+=1
	if l[4]=='Medium':
		#print 
		if medium_irmsd> float(l[2]):
			minmedium=i
			medium_fnat=float(l[1])
			medium_irmsd=float(l[2])
			medium_lrmsd=float(l[3])
		countmedium+=1
	if l[4]=='Acceptable':
		if accept_irmsd> float(l[2]):
			minaccept=i
			accept_fnat=float(l[1])
			accept_irmsd=float(l[2])
			accept_lrmsd=float(l[3])
		countaccept+=1
	i+=1
if counthigh>0:
	ret=[minhigh, high_fnat, high_irmsd, high_lrmsd, counthigh, 'High']
elif countmedium>0:
	ret=[minmedium, medium_fnat, medium_irmsd, medium_lrmsd, countmedium, 'Medium']
elif countaccept>0:
	ret=[minaccept, accept_fnat, accept_irmsd, accept_lrmsd, countaccept, 'Acceptable']
else:
	ret=[]
	'''
ret=[minp, fnat, irmsd, lrmsd, qlty]
total_good = counthigh+ countmedium+ countaccept
f.close()
'''
f=open(args.f, 'r')
j=0
for line in f:
	l=line.split()
	if l[4] in ['High', 'Medium', 'Acceptable']:
		#print line.split()[3]
		minfirst=j
		first_fnat=float(l[1])
		first_irmsd=float(l[2])
		first_lrmsd=float(l[3])
		break
	j+=1
first=[minfirst, first_fnat, first_irmsd, first_lrmsd, l[4]]
'''
#print first
print(ret)
print(total_good)
#print i


