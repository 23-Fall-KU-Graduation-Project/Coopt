import argparse
import torch
import torch.nn as nn
import os 
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.meters import get_meters,Meter,ScalarMeter,flush_scalar_meters
import sys; sys.path.append("..")
from sam import SAM
from utility.trades import AT_TRAIN, l2_norm,squared_l2_norm
from tensorboardX import SummaryWriter

global writer

def train(args,model,log,device,dataset,optimizer,train_meters,epoch,scheduler):
    model.train()
    log.train(len_dataset=len(dataset.train))

    for batch_idx, batch in enumerate(dataset.train):
        inputs, targets = (b.to(device) for b in batch)

        if args.sgd or args.adam:
            optimizer.zero_grad()
            enable_running_stats(model)
            predictions = model(inputs)
            loss = torch.nn.functional.cross_entropy(predictions, targets,reduction="none")
            train_meters["CELoss"].cache((loss.sum()/loss.size(0)).cpu().detach().numpy())
            loss.mean().backward()
            optimizer.step()
        else: # SAM
            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            train_meters["CELoss"].cache((loss.sum()/loss.size(0)).cpu().detach().numpy())
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

        with torch.no_grad():
            correct = torch.argmax(predictions.data, 1) == targets
            _, top_correct = predictions.topk(5)
            top_correct = top_correct.t()
            corrects = top_correct.eq(targets.view(1,-1).expand_as(top_correct))
            for k in range(5):
                correct_k = corrects[:k].float().sum(0)
                acc_list = list(correct_k.cpu().detach().numpy())
                train_meters["top{}_accuracy".format(k)].cache_list(acc_list)
            log(model, loss.cpu(), correct.cpu(), scheduler.lr())
            scheduler(epoch) # for default lr scheduler
            #scheduler.step() # for cosineif (batch_idx % 10) == 0:
    results = flush_scalar_meters(train_meters)
    for k, v in results.items():
        if k != "best_val":
            writer.add_scalar("train" + "/" + k, v, epoch)
    writer.add_scalar("train"+"/lr",scheduler.lr(),epoch)

def val(model,log,dataset,val_meters,optimizer,scheduler,epoch):
    if args.bilevel: # SAM + AT or SAM + TRADES, bilevel optimization
        model.eval()
        #log.eval(len_dataset=len(dataset.test))
        corrects = 0
        for batch_idx,batch in enumerate(dataset.test):
            optimizer.zero_grad()
            enable_running_stats(model)
            x_natural, y = (b.to(device) for b in batch)
            loss, loss_natural,loss_robust,adv_pred,pred= AT_TRAIN(model,args,x_natural,y,optimizer)
            val_meters["natural_loss"].cache((loss_natural).cpu().detach().numpy())
            val_meters["robust_loss"].cache((loss_robust).cpu().detach().numpy())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                #acc calculation
                adv_correct = torch.argmax(adv_pred.data,1) == y
                correct = torch.argmax(pred.data, 1) == y
                _, top_adv_correct = adv_pred.topk(5)
                _, top_correct = pred.topk(5)
                top_adv_correct = top_adv_correct.t()
                top_correct = top_correct.t()
                top_adv_corrects = top_correct.eq(y.view(1,-1).expand_as(top_adv_correct))
                corrects = top_correct.eq(y.view(1,-1).expand_as(top_correct))
                for k in range(5):
                    adv_correct_k = top_adv_corrects[:k].float().sum(0)
                    correct_k = corrects[:k].float().sum(0)
                    adv_acc_list = list(adv_correct_k.cpu().detach().numpy())
                    acc_list = list(correct_k.cpu().detach().numpy())
                    val_meters["top{}_adv_accuracy".format(k)].cache_list(adv_acc_list)
                    val_meters["top{}_accuracy".format(k)].cache_list(acc_list)
                # log(model, loss.cpu(), correct.cpu(),scheduler.lr)
            if (batch_idx % 10) == 0:
                print(
                    "Epoch: [{}][{}/{}] \t Loss {:.3f}\t Adv_Loss {:.3f}\t Acc {:.3f}\t Adv_Acc {:.3f}\t".format(
                            epoch, batch_idx, len(dataset.test), loss_natural.item(),loss_robust.item(),adv_correct.float().mean().item(),
                            correct.float().mean().item()
                        )
                )
        results = flush_scalar_meters(val_meters)
        for k, v in results.items():
            if k != "best_val":
                writer.add_scalar("val" + "/" + k, v, epoch)
        writer.add_scalar("val"+"/lr",scheduler.lr(),epoch)

    else: # Single level optimization (SAM,ADAM,SGD)
        model.eval()
        log.eval(len_dataset=len(dataset.test))

        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                val_meters["CELoss"].cache((loss.sum()/loss.size(0)).cpu().detach().numpy())
                correct = torch.argmax(predictions.data, 1) == targets
                _, top_correct = predictions.topk(5)
                top_correct = top_correct.t()
                corrects = top_correct.eq(targets.view(1,-1).expand_as(top_correct))
                for k in range(5):
                    correct_k = corrects[:k].float().sum(0)
                    acc_list = list(correct_k.cpu().detach().numpy())
                    val_meters["top{}_accuracy".format(k)].cache_list(acc_list)
                log(model, loss.cpu(), correct.cpu())
                
        results = flush_scalar_meters(val_meters)
        for k, v in results.items():
            if k != "best_val":
                writer.add_scalar("train" + "/" + k, v, epoch)
        writer.add_scalar("val"+"/lr",scheduler.lr(),epoch)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=8, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    parser.add_argument("--bilevel",action='store_true', help="use bilevel optimization to do SAM+AT. default is false")
    parser.add_argument("--trades",action="store_true",help="use trades")
    parser.add_argument("--sgd", action='store_true', help="use sgd.")
    parser.add_argument("--adam",action='store_true',help="use adam")
    args = parser.parse_args()
    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    writer = SummaryWriter(log_dir = "./samat/runs") # directory for tensorboard logs
    log_dir = "./samat/best_checkpoint" # directory for model checkpoints
    train_meters = get_meters("train",model)
    val_meters = get_meters("val",model)
    val_meters["best_val"] = ScalarMeter("best_val")
    best_val = 0.0
    base_optimizer = torch.optim.SGD
    if args.sgd: # use sgd
        optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay,momentum=args.momentum)
        print("using sgd")
    elif args.adam:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
        print("using adam")
    else:
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        print("using SAM")
    
    if args.bilevel:
        bilevel_optim = torch.optim.SGD(model.parameters(),lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        bilevel_scheduler = StepLR(bilevel_optim,args.learning_rate,args.epochs)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)
        
    for epoch in range(args.epochs):
        train(args,model,log,device,dataset,optimizer,train_meters,epoch,scheduler) # train
        
        val_meters["best_val"].cache(best_val)
        if args.bilevel:
            results = val(model,log,dataset,val_meters,bilevel_optim,bilevel_scheduler,epoch)
        else:
            results = val(model,log,dataset,val_meters,optimizer,scheduler,epoch)
        if results["top1_accuracy"] > best_val:
            best_val = results["top1_accuracy"]
        torch.save(model, os.path.join(log_dir, "best.pth"))
        
        writer.add_scalar("val/best_val", best_val, epoch)
        if epoch ==0 or (epoch+1) % 10 == 0:
                torch.save(
                    model,
                    os.path.join("./samat/checkpoint","epoch_{}.pth".format(epoch))
                )

    log.flush()
