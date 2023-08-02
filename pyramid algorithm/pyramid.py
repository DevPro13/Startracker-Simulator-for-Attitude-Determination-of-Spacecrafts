
#smart sequence of triad indices
n=5 #number of stars in the frame

for dj in range(1, n-1):
    for dk in range(1, n-dj):
        for i in range(1, n-dk-dj+1):
            j=i+dj
            k=j+dk
            print(f"[{i} {j} {k}]")

