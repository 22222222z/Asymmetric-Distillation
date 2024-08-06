import os
import time
import math
import utils1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from torch.autograd import Variable
from evaluate import evaluate, evaluate_kd
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from my_loss_function import loss_DA
from LabelSmoothing import smooth_cross_entropy_loss
import random
from openset_test import test
from augmentation_train import mix_batch, verify_teacher

# KD train and evaluate
def train_and_evaluate_kd(model, teacher_model, train_dataloader, val_dataloader, optimizer,
                       loss_fn_kd, warmup_scheduler, params, args, dataloaders, restore_file=None, writer=None):
    """
    KD Train the model and evaluate every epoch.
    """
    # reload weights from restore_file if specified
    start_epoch = 0
    params.num_epochs = args.total_epoch
    if args.resume:
        args.restore_file = 'last'
    if (args.restore_file is not None) and os.path.exists(os.path.join(args.model_dir, args.restore_file + '.pth.tar')):
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        checkpoint = utils1.load_checkpoint(restore_path, model, optimizer)
        start_epoch = checkpoint['epoch']

    # tensorboard setting
    if not writer :
        log_dir = args.model_dir + '/tensorboard/'
        writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0
    teacher_model.eval()
    # teacher_model2.eval()   #
    teacher_acc = evaluate_kd(teacher_model, val_dataloader, params)
    print(">>>>>>>>>The teacher accuracy: {}>>>>>>>>>".format(teacher_acc['accuracy']))

    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    #-------------------------
    # evaluate open set
    #-------------------------
    outloader = dataloaders['test_unknown']

    for epoch in range(start_epoch, params.num_epochs):

        if epoch > 0:   # 0 is the warm up epoch
            scheduler.step()
        logging.info("Epoch {}/{}, lr:{}".format(epoch + 1, params.num_epochs, optimizer.param_groups[0]['lr']))


        train_acc, train_loss = train_kd(model, teacher_model, optimizer, loss_fn_kd, train_dataloader, warmup_scheduler, params, args, epoch)
        # Evaluate
        val_metrics = evaluate_kd(model, val_dataloader, params)

        #------------------------------------
        # open set test
        #------------------------------------
        # results = test(model, criterion, testloader, outloader, epoch=epoch, **options)
        results = test(model, val_dataloader, outloader, epoch=epoch)

        print("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(epoch+1,
                                                                                              results['ACC'],
                                                                                              results['AUROC'],
                                                                                              results['OSCR']))
        logging.info("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(epoch+1,
                                                                                              results['ACC'],
                                                                                              results['AUROC'],
                                                                                              results['OSCR']))

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        if (epoch + 1) % 20 == 0 and args.save_interval:
            utils1.save_checkpoint_interval(epoch, 
                                {'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                                is_best=False,
                                checkpoint=args.model_dir)

        # Save weights
        if isinstance(model, torch.nn.DataParallel):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        # torch.save(model_state_dict, "model.pth")
        # if dist.get_rank() == 0:
        utils1.save_checkpoint({'epoch': epoch + 1,
                            #    'state_dict': model.state_dict(),
                            'state_dict': model_state_dict,
                            'optim_dict' : optimizer.state_dict()},
                            is_best=is_best,
                            checkpoint=args.model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            file_name = "eval_best_result.json"
            best_json_path = os.path.join(args.model_dir, file_name)
            utils1.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(args.model_dir, "eval_last_result.json")
        utils1.save_dict_to_json(val_metrics, last_json_path)

        # Tensorboard
        writer.add_scalar('Train_accuracy', train_acc, epoch)
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Test_accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Test_loss', val_metrics['loss'], epoch)
        # export scalar data to JSON for external processing
    writer.close()

def train_kd(model, teacher_model, optimizer, loss_fn_kd, dataloader, warmup_scheduler, params, args, epoch, flag=None):
    """
    KD Train the model on `num_steps` batches
    """
    
    # set model to training mode
    model.train()
    teacher_model.eval()
    # teacher_model2.eval()    #
    loss_avg = utils1.RunningAverage()
    losses = utils1.AverageMeter()
    total = 0
    correct = 0

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch, batch_idx) in enumerate(dataloader):

            logger = {"epoch": epoch, "iter": i}
            if epoch<=0:
                warmup_scheduler.step()
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
            output_batch = model(train_batch)

            output_teacher_batch = teacher_model(train_batch)
            output_teacher_batch.cuda()
            output_teacher_batch = Variable(output_teacher_batch, requires_grad=False)
            
            # loss_fn_kd = loss_DIST
            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)

            rand = random.random()
            if args.da and rand > 0.5:
                if args.cutmix:
                    option = 'cutmix'
                elif args.mixup:
                    option = 'mixup'

                lam, rand_index, aug_batch = mix_batch(train_batch, option, args, labels_batch)

                output_batch_aug = model(aug_batch)
                output_teacher_batch_aug = teacher_model(aug_batch)

                loss_mix = 0
                loss_uni = 0

                learn_list = []
                uni_list = []
                learn_list, uni_list, p_label = verify_teacher(output_teacher_batch_aug, labels_batch, labels_batch[rand_index], lam, args)
                if len(learn_list) != 0:
                    loss_mix = loss_DA(output_batch_aug[learn_list], output_teacher_batch[learn_list], params)
                if len(uni_list) != 0:
                    loss_uni = smooth_cross_entropy_loss(output_batch_aug[uni_list], labels_batch[uni_list], 0) * lam + \
                            smooth_cross_entropy_loss(output_batch_aug[uni_list], labels_batch[rand_index][uni_list], 0) * (1. - lam)
                update_logger = {'lam': lam, 'learn list': len(learn_list), 'uni_list': len(uni_list)}
                logger.update(update_logger)

                loss_mutual_information = 0
                loss_ml = loss_DA
                loss_mutual_information = loss_ml(output_batch_aug, output_teacher_batch, params) * lam + loss_ml(output_batch_aug, output_teacher_batch[rand_index], params) * (1. - lam)

                loss = loss_mix + loss_uni + loss_mutual_information
                update_logger = {'sample': '{} sample'.format(option), 'loss': "{:.2f}".format(loss.cpu().detach().numpy().tolist())}
                logger.update(update_logger)

            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            _, predicted = output_batch.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()
            # update the average loss
            loss_avg.update(loss.data)
            losses.update(loss.item(), train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()
    acc = 100.*correct/total
    logging.info("- Train accuracy: {acc:.4f}, training loss: {loss:.4f}".format(acc = acc, loss = losses.avg))
    return acc, losses.avg


# normal training
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer,
                       loss_fn, params, model_dir, warmup_scheduler, args, dataloaders, restore_file=None):
    """
    Train the model and evaluate every epoch.
    """
    # reload weights from restore_file if specified
    start_epoch = 0
    if args.resume:
        args.restore_file = 'last'
    if (args.restore_file is not None) and os.path.exists(os.path.join(args.model_dir, args.restore_file + '.pth.tar')):
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        checkpoint = utils1.load_checkpoint(restore_path, model, optimizer)
        start_epoch = checkpoint['epoch']
    # if restore_file is not None:
    #     restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    #     logging.info("Restoring parameters from {}".format(restore_path))
    #     utils1.load_checkpoint(restore_path, model, optimizer)
    # dir setting, tensorboard events will save in the dirctory
    log_dir = args.model_dir + '/base_train/'
    writer = SummaryWriter(log_dir=log_dir)

    best_val_acc = 0.0

    # learning rate schedulers
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    #-------------------------
    # evaluate open set
    #-------------------------
    outloader = dataloaders['test_unknown']

    for epoch in range(start_epoch, params.num_epochs):
        if epoch > 0:   # 1 is the warm up epoch
            scheduler.step(epoch)

        # Run one epoch
        logging.info("Epoch {}/{}, lr:{}".format(epoch + 1, params.num_epochs, optimizer.param_groups[0]['lr']))

        # compute number of batches in one epoch (one full pass over the training set)
        train_acc, train_loss = train(model, optimizer, loss_fn, train_dataloader, params, epoch, warmup_scheduler, args)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, params, args)

        #------------------------------------
        # open set test
        #------------------------------------
        # results = test(model, criterion, testloader, outloader, epoch=epoch, **options)
        results = test(model, val_dataloader, outloader, epoch=epoch)

        print("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(epoch+1,
                                                                                              results['ACC'],
                                                                                              results['AUROC'],
                                                                                              results['OSCR']))
        logging.info("Epoch {}: Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(epoch+1,
                                                                                              results['ACC'],
                                                                                              results['AUROC'],
                                                                                              results['OSCR']))

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        if (epoch + 1) % 20 == 0 and args.save_interval:
            utils1.save_checkpoint_interval(epoch, 
                                {'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                                is_best=False,
                                checkpoint=model_dir)

        # Save weights
        utils1.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                                is_best=is_best,
                                checkpoint=model_dir)
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "eval_best_results.json")
            utils1.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "eval_last_results.json")
        utils1.save_dict_to_json(val_metrics, last_json_path)

        # Tensorboard
        writer.add_scalar('Train_accuracy', train_acc, epoch)
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Test_accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Test_loss', val_metrics['loss'], epoch)
    writer.close()


# normal training function
def train(model, optimizer, loss_fn, dataloader, params, epoch, warmup_scheduler, args):
    """
    Noraml training, without KD
    """
    # set model to training mode
    model.train()
    loss_avg = utils1.RunningAverage()
    losses = utils1.AverageMeter()
    total = 0
    correct = 0
    
    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        # for i, (train_batch, labels_batch) in enumerate(tqdm(dataloader)):
        for batch_idx, (train_batch, labels_batch, i) in enumerate(tqdm(dataloader)):
            logger = {"epoch": epoch, "iter": batch_idx}
            
            if args.transform == 'augmix':
                train_batch = train_batch[0]
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
            
            if epoch<=0:
                warmup_scheduler.step()
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            optimizer.zero_grad()

            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            logging.info(logger)
            loss.backward()
            optimizer.step()

            _, predicted = output_batch.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()

            # update the average loss
            loss_avg.update(loss.data)
            losses.update(loss.data, train_batch.size(0))

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()), lr='{:05.6f}'.format(optimizer.param_groups[0]['lr']))
            t.update()

    acc = 100. * correct / total
    logging.info("- Train accuracy: {acc: .4f}, training loss: {loss: .4f}".format(acc=acc, loss=losses.avg))
    return acc, losses.avg








