from argparse import ArgumentParser, Namespace
import os
from time import time, localtime, strftime

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sys; sys.path.append("..")
from sam import SAM
from utility import AT_TRAIN, AT_VAL
from model.wideresnet import WideResNet
from data.cifar import Cifar
from utility import initialize
from utility import enable_running_stats
from utility import Meter, ScalarMeter
from utility import get_meters, flush_scalar_meters

def assert_directory(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_arguments() -> tuple[ArgumentParser, Namespace]:
    parser = ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="Use the adaptive SAM")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("--threads", default=8, type=int, help="Number of CPU threads for dataloaders")
    parser.add_argument("--rho", default=2.0, type=int, help="Parameter for SAM")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet")
    parser.add_argument("--trades",action="store_true",help="Use TRADES")
    parser.add_argument("--sgd", action='store_true', help="Use SGD.")
    parser.add_argument("--beta",default=1.0, type= float, help = "Hyperparameter for trades loss = CE + beta * adv. Range from 0.1 to 5.0")
    parser.add_argument("--gpus",default="0",type=str, help = "Gpu devices. E.g. cpu, or 0 for cuda:0")
    parser.add_argument("--step_size",default=2./255.,type = float, help = "PGD step size")
    parser.add_argument("--eps",default=8./255.,type=float,help="PGD epsilon")
    parser.add_argument("--perturb_step",default=10,type=int,help="PGD iteration step")
    parser.add_argument("--distance", default="l_inf", type=str, help="Distance norm to adversarial attack eg) l_2")

    args = parser.parse_args()
    return parser, args

def get_argument_title(parser: ArgumentParser,
                       args: Namespace
                       ) -> str:
    titles = []
    defaults = {action.dest: action.default for action in parser._actions}
    for arg in vars(args):
        value = getattr(args,arg)
        default = defaults[arg]
        if value != default:
            titles.append(f"{arg}={value}")
        elif value is True:
            titles.append(f"{arg}={value}")
    title = ",".join(titles)
    return title

def calculate_acc_adv_acc(meters, y, adv_pred, pred) -> tuple[float, float]:
    with torch.no_grad():
        adv_correct = torch.argmax(adv_pred.data,1) == y
        correct = torch.argmax(pred.data, 1) == y
        _, top_adv_correct = adv_pred.topk(5)
        _, top_correct = pred.topk(5)
        top_adv_correct = top_adv_correct.t()
        top_correct = top_correct.t()
        top_adv_corrects = top_adv_correct.eq(y.view(1,-1).expand_as(top_adv_correct))
        corrects = top_correct.eq(y.view(1,-1).expand_as(top_correct))

        for k in range(1,5):
            adv_correct_k = top_adv_corrects[:k].float().sum(0)
            correct_k = corrects[:k].float().sum(0)
            adv_acc_list = list(adv_correct_k.cpu().detach().numpy())
            acc_list = list(correct_k.cpu().detach().numpy())
            meters[f"top{k}_adv_accuracy"].cache_list(adv_acc_list)
            meters[f"top{k}_accuracy"].cache_list(acc_list)

        accuracy = correct.float().mean().item()
        adv_accuracy = adv_correct.float().mean().item()
    return accuracy, adv_accuracy

def print_progress(batch_size: int, 
                   epoch: int,
                   batch_idx: int,
                   loss_natural: float,
                   loss_robust: float,
                   accuracy: float,
                   adv_accuracy: float):
    print(" \t ".join((f"Epoch: [{epoch}][{batch_idx}/{batch_size}]",
                                  f"Loss {loss_natural:.3f}",
                                  f"Adv_Loss {loss_robust:.3f}",
                                  f"Acc {accuracy:.3f}",
                                  f"Adv_Acc {adv_accuracy:.3f}")))

def adv_learning(mode: str,
              args: Namespace,
              model: nn.Module,
              device: torch.device,
              scheduler: optim.lr_scheduler,
              data_loader: DataLoader,
              optimizer: optim.Optimizer,
              meters: dict[str, Meter],
              epoch: int,
              writer: SummaryWriter) -> None:
    if mode == "val":
        model.eval()
        torch.set_grad_enabled(False)
    for batch_idx, batch in enumerate(data_loader):
        x_natural, y = (b.to(device) for b in batch)

        if mode == "train":
            enable_running_stats(model)
            at_result = AT_TRAIN(model, device, args, x_natural, y, optimizer)
        else:
            at_result = AT_VAL(model, device, args, x_natural, y)
        _, loss_natural, loss_robust, x_adv = at_result

        meters["natural_loss"].cache(loss_natural)
        meters["robust_loss"].cache(loss_robust)

        if (batch_idx % 10) == 0:
            with torch.no_grad():
                pred = model(x_natural)
                adv_pred = model(x_adv)
            accuracy, adv_accuracy = calculate_acc_adv_acc(meters, y, adv_pred, pred)
            print_progress(len(data_loader), epoch, batch_idx,
                           loss_natural, loss_robust, accuracy, adv_accuracy)
    torch.set_grad_enabled(True)

    results = flush_scalar_meters(meters)
    for k, v in results.items():
        if k != "best_val":
            if mode == "train":
                writer.add_scalar(f"train/{k}", v, epoch)
            else:
                writer.add_scalar(f"adv_val/{k}", v, epoch)
    writer.add_scalar(f"{mode}/lr", scheduler.get_last_lr(), epoch)
    return results

def main():
    parser, args = get_arguments()
    initialize(seed=42)

    # Device
    if args.gpus != "cpu":
        device = torch.device(f"cuda:{args.gpus}")
        assert torch.cuda.is_available()
    
    # Save logs
    dir_prefix = os.path.join("..", "test")
    argument_title = get_argument_title(parser, args)
    start_time = strftime("%Y-%m-%d_%H-%M-%S", localtime(time()))
    title = argument_title + "-" + start_time
    log_dir = os.path.join(dir_prefix, "runs", title)
    checkpoint_dir = os.path.join(dir_prefix, "checkpoint", title)
    command_line_dir = os.path.join(dir_prefix, "command_lines")

    assert_directory(dir_prefix)
    assert_directory(log_dir)
    assert_directory(checkpoint_dir)
    assert_directory(command_line_dir)

    command_line_input = " ".join(sys.argv)
    with open(os.path.join(command_line_dir, title + ".txt"), "w") as file:
        file.write(command_line_input)

    writer = SummaryWriter(log_dir=log_dir)
    dataset = Cifar(args.batch_size, args.threads)
    model = WideResNet(depth=args.depth,
                       widen_factor=args.width_factor,
                       dropRate=args.dropout,
                       num_classes=10).to(device)

    # Train
    train_meters = get_meters("train", model)
    val_meters = get_meters("val", model)
    val_meters["best_val"] = ScalarMeter("best_val")
    if args.sgd:
        # AT / TRADES
        used_optimizer = "SGD"
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                    weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        # SAMAT / SAMTRADES
        used_optimizer = "SAM"
        base_optimizer = optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho,
                        adaptive=args.adaptive, lr=args.learning_rate,
                        momentum=args.momentum, weight_decay=args.weight_decay)
    print(f"using {used_optimizer}")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)

    best_val = 0.0
    for epoch in range(args.epochs):
        val_meters["best_val"].cache(best_val)

        adv_learning("train", args, model, device, scheduler, dataset.train, optimizer, train_meters, epoch, writer)
        scheduler.step()
        results = adv_learning("val", args, model, device, scheduler, dataset.test, optimizer, val_meters, epoch, writer)

        # Save best checkpoint
        if results["top1_accuracy"] > best_val:
            best_val = results["top1_accuracy"]
            torch.save(model, os.path.join(checkpoint_dir, "best.pth"))
        
        # Save checkpoint periodically
        writer.add_scalar("val/best_val", best_val, epoch)
        if (epoch == 0) or ((epoch + 1) % 10 == 0):
            torch.save(model, os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))

if __name__ == "__main__":
    main()