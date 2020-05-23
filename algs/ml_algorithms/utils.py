def compute_mse(y1,y2):
    if len(y1) != len(y2):
        raise ValueError("y1.length != y2.length")
    y1 = list(y1)
    y2 = list(y2)
    return sum([(y1[i]-y2[i])**2 for i in range(len(y1))]) / len(y1)
