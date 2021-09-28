import numpy as np

def read_gt_from_txt(path):

    with open(path) as f:
        lines = f.readlines()
        
    gt_mat = []
    i = 0
    for line in lines:
        
        line = line[:-3]
        if line.find(",")>0:
            values_str = line.split(",")
        elif line.find("\t")>0:
            values_str = line.split("\t")
#            values_str = line.split("\t\t")
        elif line.find(",")>0:
            values_str = line[i].split(",")
        else:
            values_str_ = lines[i].split("\t")
            values_str = values_str_[0].split(" ")
        
        i+=1
#        print(values_str, len(values_str))
        if len(values_str)>=0:
            try:
                values_int = [int(val) for val in values_str]
                gt_mat.append(values_int)
            except:
                values_int = [-1,-1,-1,1]
                gt_mat.append(values_int)
    
    return np.array(gt_mat)
        