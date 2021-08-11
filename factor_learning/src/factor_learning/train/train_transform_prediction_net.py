# Copyright (c) Facebook, Inc. and its affiliates.

from factor_learning.dataio.DigitImageTfPairsDataset import DigitImageTfPairsDataset
from factor_learning.utils import utils

import os

import numpy as np
import hydra
from attrdict import AttrDict

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from factor_learning.network.transform_networks import ConstantNet, TfRegrLinear, TfRegrLinearClass, TfRegrNonlinearClass, FeatMapClassNet

from datetime import datetime


class TransformRelPredictionNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # init config
        self.cfg = cfg
        self.learning_rate = self.cfg.train.learning_rate
        self.task = self.cfg.network.model

        # init model
        if (self.cfg.network.model == 'const'):
            self.model = ConstantNet()
        elif (self.cfg.network.model == 'linreg'):
            self.model = TfRegrLinear(
                input_size=2*cfg.dataloader.imgfeat_dim, output_size=4)
        elif (self.cfg.network.model == 'linregclass'):
            self.model = TfRegrLinearClass(
                input_size=2*cfg.dataloader.imgfeat_dim*self.cfg.num_classes, output_size=4)
        elif (self.cfg.network.model == 'nonlinregclass'):
            self.model = TfRegrNonlinearClass(
                input_size=2*cfg.dataloader.imgfeat_dim*self.cfg.num_classes, hidden_size=32, output_size=4)
        elif (self.cfg.network.model == 'linregfeatmap'):
            self.model = FeatMapClassNet()
        else:
            print("Model not found")
            return
        
        # init metrics
        # self.accuracy = pl.metrics.Accuracy()

    def forward(self, x, *args):
        return self.model(x, *args)
        
    # for saving torchscript model
    # def forward(self, x1, x2, k):
    #     return self.model(x1, x2, k)

    def loss_function(self, pred, gt):

        if (self.task == 'linregfeatmap'):
            loss = F.cross_entropy(pred, gt)
        else:
            if (self.cfg.train.weighted_mse_loss):
                mse_wts = torch.cuda.FloatTensor(self.cfg.train.mse_wts) if self.device == torch.device(
                    'cuda:0') else torch.FloatTensor(self.cfg.train.mse_wts)
                loss = utils.weighted_mse_loss(pred, gt, mse_wts)
            else:
                loss = F.mse_loss(pred, gt)

        return {'loss': loss}

    def normalize_feat_inputs(self, feat_i, feat_j):

        feat_i = utils.normalize_imgfeats(
            feat_i, self.cfg.transforms.norm_mean_imgfeat, self.cfg.transforms.norm_std_imgfeat, self.device)
        feat_j = utils.normalize_imgfeats(
            feat_j, self.cfg.transforms.norm_mean_imgfeat, self.cfg.transforms.norm_std_imgfeat, self.device)

        return feat_i, feat_j

    def labels2vec(self, labels):
        one_hot_vec = torch.nn.functional.one_hot(labels, self.cfg.num_classes)
        return one_hot_vec

    def training_step(self, batch, batch_idx):

        if (self.task == 'linregfeatmap'):
            img_i, img_j, pose_i, pose_j, feat_i, feat_j, featmap_i, featmap_j, metadata = batch
        else:                        
            img_i, img_j, pose_i, pose_j, feat_i, feat_j, metadata = batch

        # feat_i.squeeze_()
        # feat_j.squeeze_()

        if (self.cfg.transforms.normalize_imgfeat_inputs):
            feat_i, feat_j = self.normalize_feat_inputs(feat_i, feat_j)

        if (self.task == 'const') | (self.task == 'linreg'):
            pose_ij_pred = self.forward(feat_i, feat_j)
            pose_ji_pred = self.forward(feat_j, feat_i)
        elif (self.task == 'linregclass') | (self.task == 'nonlinregclass'):
            class_vec = self.labels2vec(metadata['class_label'])
            pose_ij_pred = self.forward(feat_i, feat_j, class_vec)
            pose_ji_pred = self.forward(feat_j, feat_i, class_vec)
        elif (self.task == 'linregfeatmap'):
            labels_gt = metadata['class_label']
            logits_pred = self.forward(featmap_i)
            # prob_pred = F.softmax(logits_pred, dim=1)
            # pose_ij_pred = self.forward(feat_i, feat_j, prob_pred)
            # pose_ji_pred = self.forward(feat_j, feat_i, prob_pred)

        pose_ij_xyh = utils.tf2d_between(pose_i, pose_j, device=self.device)
        pose_ji_xyh = utils.tf2d_between(pose_j, pose_i, device=self.device)
        pose_ij_gt = utils.tf2d_net_input(pose_ij_xyh)
        pose_ji_gt = utils.tf2d_net_input(pose_ji_xyh)

        if (self.task == 'linregfeatmap'):
            loss = self.loss_function(logits_pred, labels_gt)['loss']
            train_acc = self.accuracy(logits_pred, labels_gt)
        else:
            loss = self.loss_function(pose_ij_pred, pose_ij_gt)[
            'loss'] + self.loss_function(pose_ji_pred, pose_ji_gt)['loss']
            train_acc = None

        self.log('train_loss_step', loss)

        return {'loss': loss, 'train_acc': train_acc}

    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.Tensor([x['loss']
                                       for x in train_step_outputs]).mean()
        self.log('train_loss_epoch', avg_train_loss)

        # avg_train_acc = torch.Tensor([x['train_acc']
        #                             for x in train_step_outputs]).mean()
        # self.log('train_acc_epoch', avg_train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results['val_acc'] = results['train_acc']
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.Tensor([x['loss']
                                     for x in val_step_outputs]).mean()        
        self.log('val_loss_epoch', avg_val_loss)

        # avg_val_acc = torch.Tensor([x['val_acc']
        #                             for x in val_step_outputs]).mean()
        # self.log('val_acc_epoch', avg_val_acc, prog_bar=True)

    def data_loader(self, datatype="train"):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_list = []
        for (dataset_idx, dataset_name) in enumerate(self.cfg.dataset_names):
            srcdir_dataset = "{0}/local/datasets/{1}".format(
                BASE_PATH, dataset_name)

            metadata = AttrDict()
            if self.cfg.class_labels:
                metadata['class_label'] = self.cfg.class_labels[dataset_idx]

            dataset_list.append(DigitImageTfPairsDataset("{0}/{1}".format(srcdir_dataset, datatype), transform=transform,
                                                         imgfeat_type=self.cfg.dataloader.imgfeat_type, downsample_imgs=self.cfg.dataloader.downsample_imgs, featmap_type=None, metadata=metadata))

        dataset = ConcatDataset(dataset_list)
        dataloader = DataLoader(dataset, batch_size=self.cfg.dataloader.batch_size,
                                shuffle=self.cfg.dataloader.shuffle, num_workers=self.cfg.dataloader.num_workers)

        return dataloader

    def configure_optimizers(self):
        weight_decay = self.cfg.train.reg_weight if (
            self.cfg.train.reg_type == "l2") else 0.0
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        return optimizer


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(BASE_PATH, "config/transform_prediction.yaml")


@hydra.main(config_path=config_path, strict=False)
def main(cfg):

    # random seed
    if cfg.random_seed is not None:
        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)

    # setup logger dirs
    prefix = cfg.prefix + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    print(cfg.pretty())
    with open("{0}/runs/{1}.txt".format(BASE_PATH, prefix), "w") as f:
        print(cfg.pretty(), file=f)
    tb_logger = pl.loggers.TensorBoardLogger(
        "{0}/logs/{1}".format(BASE_PATH, prefix))

    # setup trainer
    # trainer = pl.Trainer(gpus=1, max_epochs=cfg.train.epochs, logger=tb_logger, callbacks=[
    #  EarlyStopping(monitor='val_loss_epoch')])
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.train.epochs, logger=tb_logger)

    # initialize model, dataloaders
    model = TransformRelPredictionNet(cfg)
    train_dataloader = model.data_loader("train")
    val_dataloader = model.data_loader("test")

    # run training loop
    trainer.fit(model, train_dataloader, val_dataloader)

    # save trained model
    script = model.to_torchscript()
    torch.jit.save(script, "{0}/local/{1}/{2}_tf_regr_model_ser_epoch{3:03d}.pt".format(BASE_PATH, cfg.checkpoint.path, prefix, cfg.train.epochs))
    print("Saved model as {0}/local/{1}/{2}_tf_regr_model_ser_epoch{3:03d}.pt".format(
        BASE_PATH, cfg.checkpoint.path, prefix, cfg.train.epochs))

if __name__ == '__main__':
    main()
