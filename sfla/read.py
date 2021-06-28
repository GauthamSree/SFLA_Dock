f=open('ds.txt','r')
i=0
while i<21:
	f.readline()
	i+=1
values=[]
Fnat= f.readline().split()[1]
f.readline()
IRMS=f.readline().split()[1]
LRMS=f.readline().split()[1]
Capri_quality=f.readline().split()[1]
print([Fnat, IRMS, LRMS,Capri_quality])

