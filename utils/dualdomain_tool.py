from torch.autograd import Variable
import numpy as np

def create_input(*input):
    under_img, under_k, full_img, full_k, _shotname = input
    # u_img, u_k, img, k = input

    under_img = Variable(under_img, requires_grad=True).cuda()
    under_k = Variable(under_k, requires_grad=True).cuda()
    full_img = Variable(full_img, requires_grad=False).cuda()
    full_k = Variable(full_k, requires_grad=False).cuda()
    return under_img, under_k, full_img, full_k

def create_input_allrange(*input):
    under_img_01, under_k_01, full_img_01, full_k_01, under_img, under_k, full_img, full_k = input
    # u_img, u_k, img, k = input
    under_img_01 = Variable(under_img_01, requires_grad=True).cuda()
    under_k_01 = Variable(under_k_01, requires_grad=True).cuda()
    full_img_01 = Variable(full_img_01, requires_grad=False).cuda()
    full_k_01 = Variable(full_k_01, requires_grad=False).cuda()

    under_img = Variable(under_img, requires_grad=True).cuda()
    under_k = Variable(under_k, requires_grad=True).cuda()
    full_img = Variable(full_img, requires_grad=False).cuda()
    full_k = Variable(full_k, requires_grad=False).cuda()
    return under_img_01, under_k_01, full_img_01, full_k_01, under_img, under_k, full_img, full_k

def create_input_with_edge(*input):
    u_img, u_k, img, k, edge = input
    u_img = Variable(u_img, requires_grad=True).cuda()
    u_k = Variable(u_k, requires_grad=True).cuda()
    img = Variable(img, requires_grad=False).cuda()
    k = Variable(k, requires_grad=False).cuda()
    edge = Variable(edge, requires_grad=False).cuda()
    return u_img, u_k, img, k, edge

def abs(x):
    y = x
    min = np.min(y)
    max = np.max(y)
    y = 255.0 * (y - min)/ (max - min)
    return y

def idc(x, y, mask):
    '''
        x: the undersampled kspace
        y: the restored kspace from x
        mask: the undersampling mask
        return:
    '''
    return x + y * (1 - mask)

def create_complex_value(x):
    '''
        x: (2, h, w)
        return:
            numpy, (h, w), dtype=np.complex
    '''
    result = np.zeros_like(x[0], dtype=np.complex)
    result.real = x[0]
    result.imag = x[1]
    return result