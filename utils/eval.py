import math
import numpy as np

def psnr(label, outputs, max_val=1.):
    ''' 
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    '''
    label = label.cpu().detach().numpy()      # don't need to calculate gradient for PSNR,
    outputs = outputs.cpu().detach().numpy()  # so detach from 'requires_grad = True' and convert to numpy array
    img_diff = outputs - label
    rmse = math.sqrt(np.mean(img_diff ** 2))

    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR