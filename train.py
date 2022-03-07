import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import sys
import os

from ibp_loss import *
from reader import RunReader
from logger import get_latest_run_id

from model.imp_models import MON_NAMES, DEQ_FC, DEQ_3, DEQ_7
from model.ff_models import explicit_3, explicit_7

from data import DATA_SHAPE

def get_model(normalize, args):
    if args.arch == 'DEQ_FC':
        in_feat = DATA_SHAPE[args.dataset]['in_channels']*(DATA_SHAPE[args.dataset]['shp'][0]**2)
        model = DEQ_FC(
            in_feat,
            args.out_channels,
            args.splitting,
            normalize=normalize,
            lben=args.lben,
            lben_cond=args.lben_cond,
            ibp=args.ibp,
            m=args.m,
            n_class=10,
            ibp_init=args.ibp_init)
    elif args.arch == 'DEQ_3':
        model = DEQ_3(
            DATA_SHAPE[args.dataset]['in_channels'], 
            DATA_SHAPE[args.dataset]['shp'][0],
            args.out_channels,
            args.splitting, 
            normalize=normalize, 
            lben=args.lben,
            lben_cond=args.lben_cond,
            ibp=args.ibp,
            m=args.m,
            n_class=10,
            ibp_init=args.ibp_init)
    elif args.arch == 'DEQ_7':
        model = DEQ_7(
            DATA_SHAPE[args.dataset]['in_channels'], 
            DATA_SHAPE[args.dataset]['shp'][0],
            args.splitting,
            normalize=normalize,
            lben=args.lben,
            lben_cond=args.lben_cond,
            ibp=args.ibp,
            m=args.m,
            n_class=10,
            ibp_init=args.ibp_init)
    elif args.arch == 'explicit_3':
        model = explicit_3(
            DATA_SHAPE[args.dataset]['in_channels'],
            DATA_SHAPE[args.dataset]['shp'][0],
            args.out_channels,
            n_class=10,
            normalize=normalize,
            ibp=args.ibp,
            ibp_init=args.ibp_init
            )
    elif args.arch == 'explicit_7':
        model = explicit_7(
            DATA_SHAPE[args.dataset]['in_channels'],
            DATA_SHAPE[args.dataset]['shp'][0],
            n_class=10,
            normalize=normalize,
            ibp=args.ibp,
            ibp_init=args.ibp_init,
            num_additional=args.explicit_7_additional)
    
    return model

def resume_killed(outer_dir):
    run_id = get_latest_run_id(outer_dir) - 1
    while True:
        run_reader = RunReader(os.path.join(outer_dir, 'run_%d' % (run_id)))
        ckpt = run_reader.load_checkpoint(None, latest=True)
        if ckpt is not None:
            return run_reader.run_dir
        run_id -= 1
        if run_id < 0:
            return None 

def cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor

def is_eps_warmup(curr_epoch, args):
    return curr_epoch <= args.eps_warmup + args.eps_ramp 

def get_eps(curr_epoch, batch_idx, loader_length, args):
    if not is_eps_warmup(curr_epoch, args):
        return args.eps
    else:
        total_steps = args.eps_ramp*loader_length
        ratio = 0.25
        beta = 4
        mid_step = int(ratio*total_steps) # when we switch from polynomial to linear
        curr_step = (curr_epoch - 1 - args.eps_warmup)*loader_length + batch_idx
        t = mid_step**(beta - 1)
        alpha = args.eps/((total_steps - mid_step)*beta*t + mid_step*t)
        if curr_step < mid_step:
            return alpha*(curr_step)**beta # polynomial regime
        else:
            mid_val = alpha*(mid_step)**beta 
            return mid_val + (args.eps - mid_val)*(curr_step - mid_step)/(total_steps - mid_step)

def get_optim(model, args):
    print('Using Adam optimizer.')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return optimizer 

def get_lr_lambda(anneal_at, anneal_factor):
    def f(epoch):
        fac = 1
        for ep in anneal_at:
            if epoch > ep:
                fac *= anneal_factor
        return fac
    return f

def set_m(model, eps, args):
    m = args.m + (args.m_init - args.m)*(1 - eps/args.eps)
    model.mon.linear_module.m = m 

def train(train_loader, test_loader, model, log_writer, args): 
    optimizer = get_optim(model, args)

    lr_scheduler = None
    if args.lr_mode == 'step':
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda(args.step, args.anneal_factor))
    elif args.lr_mode != 'constant':
        raise Exception('lr mode one of constant, step')

    model = cuda(model)

    if args.resume:
        run_reader = RunReader(args.resume)
        ckpt, _ = run_reader.load_checkpoint(
            None, 
            latest=True)

        if args.lr_mode == 'step':
            model_state_dict, optim_state_dict, start_epoch, tot_iter, scheduler_state_dict = ckpt
            lr_scheduler.load_state_dict(scheduler_state_dict)
        else:
            model_state_dict, optim_state_dict, start_epoch, tot_iter = ckpt
        optimizer.load_state_dict(optim_state_dict)
        model.load_state_dict(model_state_dict)
    else:
        start_epoch = 1
        tot_iter = 0

    metrics_to_use = ['err', 'ce_loss'] if not args.ibp else ['err', 'ce_loss', 'ibp_err', 'ibp_loss']
    for metric in metrics_to_use:
        log_writer.add_metric(metric, metric_type='avg')

    train_items = ['err', 'ce_loss', 'epoch', 'iter', 'lr', 'fwd_iters', 'bkwd_iters', 'fwd_time', 'bkwd_time']
    test_items = ['ce_loss','err','epoch','iter']

    if args.ibp:
        train_items += ['ibp_err', 'ibp_loss', 'eps']
        test_items += ['ibp_err', 'ibp_loss']

    log_writer.create_data_dict(
        train_items, dict_id='train')
    log_writer.create_data_dict(
        test_items, dict_id='test')
    log_writer.create_data_dict(
        model.norm_names, dict_id='model_norm')

    scatter_mat = compute_sa(n_class=10)

    for epoch in range(start_epoch, 1 + args.epochs):
        model.train()
        start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            eps = get_eps(epoch, batch_idx, len(train_loader), args)
            if args.anneal_m:
                set_m(model, eps, args)
            if (batch_idx  == 30 or batch_idx == int(len(train_loader)/2)) and args.tune_alpha and args.arch in MON_NAMES:
                run_tune_alpha(model, cuda(batch[0]), args.max_alpha)

            data, target = cuda(batch[0]), cuda(batch[1])
            optimizer.zero_grad()
            if not args.ibp:
                preds, _ = model(data)
                ibp_preds = None
                ce_loss = nn.CrossEntropyLoss()(preds, target)
                ce_loss.backward()
                ibp_loss = None,
                eps = 0
            else:
                preds, z = model(data, eps=eps)
                ibp_preds = compute_ibp_elide_z(model.Wout, z[1], z[2], target, scatter_mat, n_class=10)
                ce_loss = nn.CrossEntropyLoss()(preds, target)
                ibp_loss = nn.CrossEntropyLoss()(ibp_preds, target)
                ibp_loss.backward()
            log_train_step(
                log_writer,
                model,
                optimizer,
                epoch,
                tot_iter, 
                preds,
                target,
                ce_loss,
                ibp_preds,
                ibp_loss,
                eps,
                args)

            log_weight_norm(log_writer, model, epoch, tot_iter, args)

            if args.grad_clip:
                for groupi in optimizer.param_groups:
                    nn.utils.clip_grad_norm_(groupi['params'], max_norm=args.grad_clip)
            optimizer.step()
            tot_iter += 1


        if args.lr_mode == 'step':
            lr_scheduler.step()

        print("Tot train time: {}".format(time.time() - start))

        val(test_loader, model, log_writer, epoch, tot_iter, args)

        if epoch % args.ckpt_every == 0:
            save_ckpt(model, optimizer, epoch, tot_iter, lr_scheduler, log_writer, args, is_latest=False) 
        save_ckpt(model, optimizer, epoch, tot_iter, lr_scheduler, log_writer, args, is_latest=True)
        sys.stdout.flush()

def save_ckpt(model, optimizer, epoch, iters, scheduler, log_writer, args, is_latest=False):
    save_items = [model.state_dict(), optimizer.state_dict(), epoch + 1, iters]
    if args.lr_mode == 'step':
        save_items += [scheduler.state_dict()]
    log_writer.ckpt_model(save_items, str(epoch), is_latest=is_latest)

def val(test_loader, model, log_writer, epoch, tot_iter, args):
    # testing code
    start = time.time()
    test_loss = 0
    test_ibp_loss = 0
    incorrect = 0
    ibp_inc = 0

    scatter_mat = compute_sa(n_class=10)

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            data, target = cuda(batch[0]), cuda(batch[1])
            preds, z = model(data, eps=args.test_eps)
            if args.ibp:
                ibp_preds = compute_ibp_elide_z(model.Wout, z[1], z[2], target, scatter_mat, n_class=10)
                ibp_loss = nn.CrossEntropyLoss(reduction='sum')(ibp_preds, target)
                test_ibp_loss += ibp_loss.item()
                ibp_inc += ibp_preds.float().argmax(1).ne(target.data).sum()
            ce_loss = nn.CrossEntropyLoss(reduction='sum')(preds, target)
            test_loss += ce_loss.item()
            incorrect += preds.float().argmax(1).ne(target.data).sum()
        nTotal = len(test_loader.dataset)
        test_loss /= nTotal
        err = 100. * incorrect.item() / float(nTotal)
        if args.ibp:
            ibp_err = 100 * ibp_inc.item() / float(nTotal)
            test_ibp_loss /= nTotal
        if not args.ibp:
            log_str = '\n\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.2f}%)'.format(
                test_loss, incorrect, nTotal, err)
            print(log_str)
            log_writer.log(log_str)
        else:
            log_str = '\n\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.2f}%), IBP Error: {}/{} ({:.2f}%)'.format(
                test_loss, 
                incorrect, nTotal, err,
                ibp_inc, nTotal, ibp_err)
            print(log_str)
            log_writer.log(log_str)

    update_dict = {
        'ce_loss' : test_loss,
        'err' : err,
        'epoch' : epoch,
        'iter' : tot_iter
    }

    if args.ibp:
        update_dict.update({
            'ibp_loss' : test_ibp_loss,
            'ibp_err' : ibp_err
            })
    log_writer.update_data_dict(
        update_dict, dict_id='test')
    log_writer.save_data_dict(
        dict_id='test')
    print("Tot test time: {}\n\n\n\n".format(time.time() - start))

def log_train_step(
    log_writer, 
    model, 
    optimizer, 
    epoch,
    tot_it, 
    preds, 
    target, 
    ce_loss, 
    ibp_preds, 
    ibp_loss,
    eps,
    args):
    incorrect = preds.float().argmax(1).ne(target.data).sum()
    err = 100. * incorrect.float() / float(len(target))
    log_writer.update_metric('err', err.item())
    log_writer.update_metric('ce_loss', ce_loss.item())

    if args.ibp:
        ibp_inc = ibp_preds.float().argmax(1).ne(target.data).sum()
        ibp_err = 100. * ibp_inc.float() / float(len(target))
        log_writer.update_metric('ibp_err', ibp_err.item())
        log_writer.update_metric('ibp_loss',  ibp_loss.item())

    if (tot_it > 0 and tot_it % args.log_every == 0):
        metrics_to_use = ['err', 'ce_loss'] if not args.ibp else ['err', 'ce_loss', 'ibp_err', 'ibp_loss']

        update_dict = {key : log_writer.get_metric(key) for key in metrics_to_use}
        update_dict.update(
            {
            'epoch' : epoch,
            'iter' : tot_it, 
            'lr' : optimizer.param_groups[0]['lr']
            })
        if args.arch in MON_NAMES:
            update_dict.update({
                'fwd_iters' : model.mon.stats.fwd_iters.val,
                'bkwd_iters' : model.mon.stats.bkwd_iters.val,
                'fwd_time' : model.mon.stats.fwd_time.val,
                'bkwd_time' : model.mon.stats.bkwd_time.val
                })
        if args.ibp:
            update_dict.update({
                'eps' : eps
                })

        log_str = 'Curr iter: {:.6f}\tLoss: {:.4f}\tError: {:.2f}\t'.format(
            tot_it, update_dict['ce_loss'], update_dict['err'])

        if args.ibp:
            log_str += 'IBP Loss: {:.4f}\tIBP Error: {:.2f}\t'.format(
                update_dict['ibp_loss'], update_dict['ibp_err'])

        if args.arch in MON_NAMES:
            model.mon.stats.report()
            model.mon.stats.reset()

        log_writer.reset_metrics()
        log_writer.update_data_dict(update_dict, dict_id='train')
        log_writer.save_data_dict(dict_id='train')
        log_writer.log(log_str)
        print(log_str)

def log_weight_norm(log_writer, model, epoch, tot_it, args):
    if (tot_it > 0 and tot_it % args.log_every == 0):
        update_dict = model.get_norms()
        log_str = 'Model norms: ' + ' '.join(['{} : {:.4f}'.format(key, val) for key, val in update_dict.items()])
        update_dict.update(
            {
            'epoch' : epoch,
            'iter' : tot_it
            }
        )
        log_writer.update_data_dict(update_dict, dict_id='model_norm')
        log_writer.save_data_dict(dict_id='model_norm')
        log_writer.log(log_str)
        print(log_str)

def run_tune_alpha(model, x, max_alpha):
    print("----tuning alpha----")
    print("current: ", model.mon.alpha)
    orig_alpha  =  model.mon.alpha
    model.mon.stats.reset()
    model.mon.alpha = max_alpha
    with torch.no_grad():
        model(x)
    iters = model.mon.stats.fwd_iters.val
    model.mon.stats.reset()
    iters_n = iters
    print('alpha: {}\t iters: {}'.format(model.mon.alpha, iters_n))
    while model.mon.alpha > 1e-4 and iters_n <= iters:
        model.mon.alpha = model.mon.alpha/2
        with torch.no_grad():
            model(x)
        iters = iters_n
        iters_n = model.mon.stats.fwd_iters.val
        print('alpha: {}\t iters: {}'.format(model.mon.alpha, iters_n))
        model.mon.stats.reset()

    if iters==model.mon.max_iter:
        print("none converged, resetting to current")
        model.mon.alpha=orig_alpha
    else:
        model.mon.alpha = model.mon.alpha * 2
        print("setting to: ", model.mon.alpha)
    print("--------------\n")


