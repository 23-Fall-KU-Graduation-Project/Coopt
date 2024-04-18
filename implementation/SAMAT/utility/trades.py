from argparse import Namespace

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
from utility.smooth_crossentropy import smooth_crossentropy

def get_adversarial_examples(model: nn.Module,
                             device: torch.device,
                             is_trades: bool,
                             distance: str,
                             perturb_step: int,
                             step_size: int,
                             epsilon: float,
                             x_natural: Tensor,
                             y: Tensor,
                             batch_size: int):
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()

    if distance == 'l_inf':
        for _ in range(perturb_step):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if is_trades:
                    loss = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(model(x_natural), dim=1),
                                    reduction='sum')
                else:
                    loss = smooth_crossentropy(model(x_adv),y,0)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_step * 2)

        for _ in range(perturb_step):
            adv = x_natural + delta

            # Optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                if is_trades: # why -1?
                    loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1),
                                           reduction='sum')
                else:
                    loss = (-1) * F.cross_entropy(model(x_adv),y)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.add_(x_natural)
            delta.clamp_(0, 1).sub_(x_natural)
            delta.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = x_natural + delta
        x_adv.requires_grad = False
    else:
        # Blackbox attack
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def AT_TRAIN(model: nn.Module,
             device: torch.device,
             args: Namespace,
             x_natural: Tensor,
             y: Tensor,
             optimizer: optim.Optimizer
             ) -> tuple[float, float, float, Tensor]:
    # ASAMT
    # x_adv = get_adversarial_examples(model, device, args.trades, args.distance,
    #                                 args.perturb_step, args.step_size, args.eps,
    #                                 x_natural, y, batch_size=len(x_natural))
    # x_adv.requires_grad = False
        
    model.train()
    optimizer.zero_grad()
    predictions = model(x_natural) 
    if args.sgd:
        if args.trades:
            # TRADES
            loss_natural =smooth_crossentropy(predictions,y)
            loss_robust = args.beta * F.kl_div(F.log_softmax(model(x_adv),dim=1),
                                               F.softmax(model(x_natural),dim=1),
                                               reduction='batchmean')
        else:
            # AT
            loss_natural = torch.tensor(0.)
            loss_robust = F.cross_entropy(model(x_adv),y)
        loss = loss_natural + loss_robust
        loss.backward()
        optimizer.step()
    else: # SAM
        # First forward-backward step to climb to local maxima
        loss_natural = smooth_crossentropy(predictions, y)
        loss_natural.backward() 
        optimizer.first_step(zero_grad=True)
        #SAMAT
        x_adv = get_adversarial_examples(model, device, args.trades, args.distance,
                                    args.perturb_step, args.step_size, args.eps,
                                    x_natural, y, batch_size=len(x_natural))
        x_adv.requires_grad = False
        model.train()
        if args.trades:
            # SAMTRADES
            loss_robust = args.beta * F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                               F.softmax(model(x_natural), dim=1),
                                               reduction='batchmean')
        else:
            # SAMAT
            loss_robust = smooth_crossentropy(model(x_adv),y)

        # Second forward-backward step
        loss_sam =smooth_crossentropy(model(x_natural), y)
        loss = loss_sam + loss_robust
        loss.backward()
        optimizer.second_step(zero_grad=True)
    
    return loss.item(), loss_natural.item(), loss_robust.item(), x_adv

def AT_VAL(model: nn.Module,
           device: torch.device,
           args: Namespace,
           x_natural: Tensor,
           y: Tensor
           ) -> tuple[float, float, float, Tensor]:
    x_adv = get_adversarial_examples(model, device, False, args.distance,
                                     args.perturb_step, args.step_size, args.eps,
                                     x_natural, y, batch_size=len(x_natural))
    
    # calculate robust loss
    predictions = model(x_natural)
    loss_natural = F.cross_entropy(predictions, y)
    if args.trades:
        loss_robust = args.beta * F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(model(x_natural), dim=1),
                                           reduction='batchmean')
    else:
        loss_robust = F.cross_entropy(model(x_adv),y)
    loss = loss_natural + loss_robust
    return loss.item(), loss_natural.item(), loss_robust.item(), x_adv
