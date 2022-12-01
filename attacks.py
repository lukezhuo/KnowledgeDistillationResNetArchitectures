import torch
import torch.nn as nn
import torch.nn.functional as F

def random_noise_attack(model, device, dat, eps, min_val=0, max_val=1):
    # Add uniform random noise in [-eps,+eps]
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    # Clip the perturbed datapoints to ensure we are in bounds 
    x_adv = torch.clamp(x_adv.clone().detach(), min_val, max_val)
    # Return perturbed samples
    return x_adv

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start, min_val=0, max_val=1):
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool

    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach()

    # If rand_start is True, add uniform noise to the sample within [-eps,+eps],
    # else just copy x_nat
    if rand_start:
        x_proj = random_noise_attack(model, device, dat, eps)
        x_proj = x_proj.clamp(min_val, max_val)
    else:
    # Make sure the sample is projected into original distribution bounds
        x_proj = x_nat.clamp(min_val, max_val)
    
    # Iterate over iters
    for i in range(iters):
        # Compute gradient w.r.t. data (we give you this function, but understand it)
        grad = gradient_wrt_data(model, device, x_proj.clone().detach(), lbl)
        # Perturb the image using the gradient
        perturb = alpha * grad.sign()
        # Clip the perturbed datapoints to ensure we still satisfy L_infinity constraint
        x_proj += perturb
        x_proj = torch.max(torch.min(x_proj.clone().detach(), x_nat + eps), x_nat - eps)
        # Clip the perturbed datapoints to ensure we are in bounds 
        x_proj = torch.clamp(x_proj, min_val, max_val)
    # Return the final perturbed samples
    return x_proj


def FGSM_attack(model, device, dat, lbl, eps, min_val=0, max_val=1):
    # - Dat and lbl are tensors
    # - eps is a float

    return PGD_attack(model, device, dat, lbl, eps, eps, 1, False, min_val, max_val)


def rFGSM_attack(model, device, dat, lbl, eps, min_val=0, max_val=1):
    # - Dat and lbl are tensors
    # - eps is a float

    return PGD_attack(model, device, dat, lbl, eps, eps, 1, True, min_val, max_val)

