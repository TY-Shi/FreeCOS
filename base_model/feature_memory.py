"""
Implementation for the Memory Bank for pixel-level feature vectors
"""

import torch
import torch.nn as nn
import numpy as np
import random


class FeatureMemory:
    def __init__(self, queue_len=500):
        # self.num_samples = num_samples
        self.queue_len = queue_len
        self.memory = None
        '''
        if dataset == 'cityscapes': # usually all classes in one image
            self.per_class_samples_per_image = max(1, int(round(memory_per_class / num_samples)))
        elif dataset == 'pascal_voc': # usually only around 3 classes on each image, except background class
            self.per_class_samples_per_image = max(1, int(n_classes / 3 * round(memory_per_class / num_samples)))
        '''

    def add_features_from_sample_learned(self, features, pred_bg):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features = features.detach()
        bs = features.shape[0]
        dim = features.shape[1]
        print("feature",features.shape)#5 64 512 512
        features = features.view(bs, dim, -1).permute(1, 0, 2)
        print("feature",features.shape)# 64 5 262144
        pred_bg = pred_bg.squeeze().view(bs, -1)
        print("pred_bg",pred_bg.shape) # 5 262144
        features = nn.functional.normalize(features, dim=0)
        print("feature",features.shape)# 64 5 262144
        print("features[:, pred_bg == 1]",features[:, pred_bg == 1].shape) # 64 1310720
        print("features[:, pred_bg == 1].mean(1)",features[:, pred_bg == 1].mean(1).shape)# 64
        new_fea = features[:, pred_bg == 1].mean(1).unsqueeze(0)
        print("new_fea",new_fea.shape) #1 64
        with torch.no_grad():
            if self.memory is None:  # was empy, first elements
                self.memory = new_fea
            else:
                self.memory = torch.cat((new_fea, self.memory), dim=0)[:self.queue_len, :]

    def get_prototype(self):
        # if self.memory is None:
        # return None
        return self.memory.mean(0).unsqueeze(0)


class FeatureMemory_TWODomain_NC:
    def __init__(self, queue_len=10000):
        # self.num_samples = num_samples
        self.queue_len = queue_len
        self.memory_neglabel = None

    def deque_memory(self, memory, new_fea):
        with torch.no_grad():
            if memory is None and new_fea is not None:  # was empy, first elements
                memory = new_fea.detach()
            elif memory is not None and new_fea is not None:
                #print("new_fea",new_fea.shape)
                memory = torch.cat((new_fea, memory), dim=0)[:self.queue_len, :]
        return memory

    def add_features_from_sample_learned(self,negfeature_label):
        """
        Updates the memory bank with some quality feature vectors per class
        feature:N,C

        """
        # pos_label = posfeature_label.detach()
        # neg_label = negfeature_label.detach()
        # pos_unlabel = posfeature_unlabel.detach()
        # neg_unlabel = negfeature_unlabel.detach()
        self.memory_neglabel = self.deque_memory(self.memory_neglabel, negfeature_label)
        #print("self.memory_poslabel",self.memory_poslabel.shape)
        #print("self.memory_posunlabel",self.memory_posunlabel.shape)



class FeatureMemory_dual:
    def __init__(self, queue_len=500):
        # self.num_samples = num_samples
        self.queue_len = queue_len
        self.memory = [None] * 2
        '''
        if dataset == 'cityscapes': # usually all classes in one image
            self.per_class_samples_per_image = max(1, int(round(memory_per_class / num_samples)))
        elif dataset == 'pascal_voc': # usually only around 3 classes on each image, except background class
            self.per_class_samples_per_image = max(1, int(n_classes / 3 * round(memory_per_class / num_samples)))
        '''

    def add_features_from_sample_learned(self, features_s, features_t, pred_bg):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features_s = features_s.detach()
        features_t = features_t.detach()
        bs = features_s.shape[0]
        dim = features_s.shape[1]
        features_s = features_s.view(bs, dim, -1).permute(1, 0, 2)
        features_t = features_t.view(bs, dim, -1).permute(1, 0, 2)
        pred_bg = pred_bg.squeeze().view(bs, -1)
        features_s = nn.functional.normalize(features_s, dim=0)
        features_t = nn.functional.normalize(features_t, dim=0)
        new_fea_s = features_s[:, pred_bg == 1].mean(1).unsqueeze(0)
        new_fea_t = features_t[:, pred_bg == 1].mean(1).unsqueeze(0)
        with torch.no_grad():
            if self.memory[0] is None:  # was empy, first elements
                self.memory[0] = new_fea_s
            else:
                self.memory[0] = torch.cat((new_fea_s, self.memory[0]), dim=0)[:self.queue_len, :]
            if self.memory[1] is None:  # was empy, first elements
                self.memory[1] = new_fea_t
            else:
                self.memory[1] = torch.cat((new_fea_t, self.memory[1]), dim=0)[:self.queue_len, :]

    def get_prototype(self):
        # if self.memory is None:
        # return None
        return self.memory[0].mean(0).unsqueeze(0), self.memory[1].mean(0).unsqueeze(0)


class FeatureMemory_5class:
    def __init__(self, queue_len=100):
        # self.num_samples = num_samples
        self.queue_len = queue_len
        self.memory = [None] * 5

    def de_and_enqueue(self, keys, vals, cat):  # keys: new_fea shape: n(<5),64
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]  # 64
        # batch_size = bs
        if self.memory[cat] is None:  # was empy, first elements
            self.memory[cat] = keys.unsqueeze(0)
        else:
            self.memory[cat] = torch.cat((keys.unsqueeze(0), self.memory[cat]), dim=0)[:self.queue_len, :]

    def construct_region(self, fea, pred, task_ids):
        bs = fea.shape[0]
        dim = fea.shape[1]
        pred = pred.squeeze().view(bs, -1)
        for b in range(bs):
            pos = torch.as_tensor(task_ids[b] + 1)
            neg = torch.as_tensor(0)
            pred[b, :] = torch.where(pred[b, :] == 1, pos.cuda(), neg.cuda())
        val = torch.unique(pred)
        # print(pred.shape,val)
        # fea = fea.squeeze()
        fea = fea.view(bs, dim, -1).permute(1, 0, 2)
        new_fea = fea[:, pred == val[0]].mean(1).unsqueeze(0)  # background
        for i in val[1:]:
            if (i < 5):
                class_fea = fea[:, pred == i].mean(1).unsqueeze(0)  # 1*64
                new_fea = torch.cat((new_fea, class_fea), dim=0)
        val = torch.tensor([i for i in val if i < 5])
        return new_fea, val.cuda()

    def add_features_from_sample_learned_fg(self, fea, label, task_ids):
        fea = fea.detach()
        label = label.detach()
        keys, vals = self.construct_region(fea, label, task_ids)
        keys = nn.functional.normalize(keys, dim=1)
        for i in range(1, 5):
            with torch.no_grad():
                self.de_and_enqueue(keys, vals, i)

    '''
    def add_features_from_sample_learned_fg(self, fea, label, task_ids):
        fea = fea.detach()
        bs = fea.shape[0]
        dim = fea.shape[1]
        fea = fea.view(bs,dim,-1).permute(1,0,2)
        fea = nn.functional.normalize(fea, dim = 0)
        pred = label.squeeze().view(bs,-1)
        for b in range(bs):
            pos = torch.as_tensor(task_ids[b]+1)
            neg = torch.as_tensor(0)
            pred[b,:] = torch.where(pred[b,:] == 1, pos.cuda(), neg.cuda())
        val = torch.unique(pred)
        #print(pred.shape,val)
        new_fea = fea[:,pred==val[0]].mean(1).unsqueeze(0)  #background
        for i in val[1:]:
            if(i<5):
                class_fea = fea[:,pred==i].mean(1).unsqueeze(0)  #1*64
                new_fea = torch.cat((new_fea,class_fea),dim = 0)
        val = torch.tensor([i for i in val if i<5])
        with torch.no_grad():
            for i in range(1,5):
                self.de_and_enqueue(new_fea,val,i)
    '''

    def add_features_from_sample_learned_bg(self, features, pred_bg):
        features = features.detach()
        bs = features.shape[0]
        dim = features.shape[1]
        features = features.view(bs, dim, -1).permute(1, 0, 2)
        pred_bg = pred_bg.squeeze().view(bs, -1)
        features = nn.functional.normalize(features, dim=0)
        new_fea = features[:, pred_bg == 1].mean(1).unsqueeze(0)
        with torch.no_grad():
            if self.memory[0] is None:  # was empy, first elements
                self.memory[0] = new_fea
            else:
                self.memory[0] = torch.cat((new_fea, self.memory[0]), dim=0)[:self.queue_len, :]

    def get_prototype(self):
        # if self.memory is None:
        # return None
        return [self.memory[0].mean(0).unsqueeze(0), self.memory[1].mean(0).unsqueeze(0),
                self.memory[2].mean(0).unsqueeze(0), self.memory[3].mean(0).unsqueeze(0),
                self.memory[4].mean(0).unsqueeze(0)]


if __name__ == '__main__':
    # x = torch.randn((5, 64, 512, 512))
    # x_bg = torch.ones((5,512,512))
    # feature_memory = FeatureMemory()
    # feature_memory.add_features_from_sample_learned(x,x_bg)
    # feature_memory.add_features_from_sample_learned(x, x_bg)
    # feature_memory.add_features_from_sample_learned(x, x_bg)
    # print("feature_memory",feature_memory.memory.shape)
    positive_key = torch.randn((100, 64))
    negative_keys = torch.randn((10, 64))
    positive_key = torch.nn.functional.normalize(positive_key, dim=-1)
    negative_keys = torch.nn.functional.normalize(negative_keys, dim=-1)
    positive_logit = torch.sum(positive_key * positive_key, dim=1, keepdim=True)
    negative_logits = positive_key @ negative_keys.T
    print("positive_key * positive_key",(positive_key * positive_key).shape)
    print("positive_logit",positive_logit.shape)#100 1
    print("negative_logits",negative_logits.shape) #100 10
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long)
    print("labels",labels.shape) #100
    loss = torch.nn.functional.cross_entropy(logits / 0.01, labels)
    print("logits",logits.shape)#100 11
    print("len(logits)",len(logits))
    print("loss",loss)
