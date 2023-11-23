n=int(input("Number of stars in frame"))
if n<3:
    #number of stars in frame is less than 3
    #can't make a traingle so cant compute using pyramid algo
    print("Failed, error")
elif n==3:
    #number of stars in frame is 3, it forms only one traingle
    #check for the unique solution
    print("sequence: [1 2 3]")
else:
    #number of stars in frame is more than 3
    #smart sequence of triad indices
    for dj in range(1, n-1):
        for dk in range(1, n-dj):
            for i in range(1, n-dk-dj+1):
                j=i+dj
                k=j+dk
                print(f"[{i} {j} {k}]")

def smart_triad(n):
    #number of stars in the frame is more than 3
    if(n>3):
        #smart sequence of triad indices
        for dj in range(1, n-1):
            for dk in range(1, n-dj):
                for i in range(1, n-dk-dj+1):
                    j=i+dj
                    k=j+dk
                    print(f"[{i} {j} {k}]")