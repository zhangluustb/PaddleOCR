# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# The code is refer from: https://github.com/open-mmlab/mmocr/blob/main/mmocr/core/evaluation/kie_metric.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle

__all__ = ['KIEMetric']


class KIEMetric(object):
    def __init__(self, main_indicator='hmean', **kwargs):
        self.main_indicator = main_indicator
        self.reset()
        self.node = []
        self.gt = []
        self.preds = []
        self.batchs = []

    def __call__(self, preds, batch, **kwargs):
        nodes, edges = preds
        gts, tag = batch[4].squeeze(0), batch[5].tolist()[0]
        gts = gts[:tag[0], :1].reshape([-1])
        self.node.append(nodes.numpy())
        self.gt.append(gts)
        
        self.preds.append(preds)
        self.batchs.append(batch)
        # result = self.compute_f1_score(nodes, gts)
        # self.results.append(result)
    '''
    copy kie loss code
    '''
    def pre_process(self, gts, tag):
        gts, tag = gts, tag.tolist()
        temp_gts = []
        batch = len(tag)
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            temp_gts.append(
                paddle.to_tensor(
                    gts[i, :num, :num + 1], dtype='int64'))
        return temp_gts

    def accuracy(self, pred, target, topk=1, thresh=None):
        """Calculate accuracy according to the prediction and target.

        Args:
            pred (torch.Tensor): The model prediction, shape (N, num_class)
            target (torch.Tensor): The target of each prediction, shape (N, )
            topk (int | tuple[int], optional): If the predictions in ``topk``
                matches the target, the predictions will be regarded as
                correct ones. Defaults to 1.
            thresh (float, optional): If not None, predictions with scores under
                this threshold are considered incorrect. Default to None.

        Returns:
            float | tuple[float]: If the input ``topk`` is a single integer,
                the function will return a single float as accuracy. If
                ``topk`` is a tuple containing multiple integers, the
                function will return a tuple containing accuracies of
                each ``topk`` number.
        """
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
            return_single = True
        else:
            return_single = False

        maxk = max(topk)
        if pred.shape[0] == 0:
            accu = [pred.new_tensor(0.) for i in range(len(topk))]
            return accu[0] if return_single else accu
        pred_value, pred_label = paddle.topk(pred, maxk, axis=1)
        pred_label = pred_label.transpose(
            [1, 0])  # transpose to shape (maxk, N)
        correct = paddle.equal(pred_label,
                               (target.reshape([1, -1]).expand_as(pred_label)))
        res = []
        for k in topk:
            correct_k = paddle.sum(correct[:k].reshape([-1]).astype('float32'),
                                   axis=0,
                                   keepdim=True)
            res.append(
                paddle.multiply(correct_k,
                                paddle.to_tensor(100.0 / pred.shape[0])))
        return res[0] if return_single else res

    def forward(self, pred, batch):
        node_preds = np.concatenate([p[0] for p in pred], 0)
        edge_preds = np.concatenate([p[1] for p in pred], 0)
        node_gts = np.concatenate([p[4] for p in batch], 0)
        tags = np.concatenate([p[5] for p in batch], 0)
        gts = self.pre_process(node_gts, tags)
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].reshape([-1]))
        node_gts = paddle.concat(node_gts)
        edge_gts = paddle.concat(edge_gts)
        if (node_gts != 0).sum().item()==0:
            return None
        node_valids = paddle.nonzero(node_gts != 0).reshape([-1])
        edge_valids = paddle.nonzero(edge_gts != -1).reshape([-1])

        return dict(
            acc_node=self.accuracy(
                paddle.gather(paddle.to_tensor(node_preds), node_valids),
                paddle.gather(node_gts, node_valids)).item(),
            acc_edge=self.accuracy(
                paddle.gather(paddle.to_tensor(edge_preds), edge_valids),
                paddle.gather(edge_gts, edge_valids)).item())

    
    
    def compute_f1_score(self, preds, gts):
        acc = ((preds.argmax(1)==gts)[np.nonzero(gts)[0]]).sum()/(np.nonzero(gts)[0]).shape[0] # without other acc
        acc_other = ((preds.argmax(1)==gts)[np.where(gts==0)[0]]).sum()/(np.where(gts==0)[0]).shape[0]
        # ignores = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25] #ignore key
        ignores = [0] #ignore other
        C = preds.shape[1]
        classes = np.array(sorted(set(range(C)) - set(ignores)))
        hist = np.bincount(
            (gts * C).astype('int64') + preds.argmax(1), minlength=C
            **2).reshape([C, C]).astype('float32')
        diag = np.diag(hist)
        recalls = diag / hist.sum(1).clip(min=1)
        precisions = diag / hist.sum(0).clip(min=1)
        f1 = 2 * recalls * precisions / (recalls + precisions).clip(min=1e-8)
        return f1[classes],acc,acc_other

    def combine_results(self, results):
        node = np.concatenate(self.node, 0)
        gts = np.concatenate(self.gt, 0)
        results,acc,acc_other = self.compute_f1_score(node, gts)
        data = {'hmean': results.mean(),"acc":acc,"acc_other":acc_other}
        return data
    
    def combine_node_edge(self):
        data = self.forward(self.preds, self.batchs)
        return data

    def get_metric(self):

        metrics = self.combine_results(self.results)
        metrics_new = self.combine_node_edge()
        metrics_new.update(metrics)
        self.reset()
        return metrics_new

    def reset(self):
        self.results = []  # clear results
        self.node = []
        self.gt = []
        self.preds = []
        self.batchs = []
