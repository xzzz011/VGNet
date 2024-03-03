import numpy as np
import heapq
import pandas as pd
from numpy import log2


class MultiModalTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train(self, epochs):
        print('start train...')
        self.model.train()

        for epoch in range(epochs):
            filenames, save_labels, save_features = [], [], []
            db_per_class_cnt = np.zeros(30)
            for data in self.train_loader:
                self.optimizer.zero_grad()
                out, feature, cor = self.model(data)
                labels = data[0][0:8]
                loss = self.loss_fn(out, labels) + cor
                loss.backward()
                self.optimizer.step()

                for i in range(int(feature.shape[0])):
                    save_labels.append(data[0][i].item())
                    save_features.append(feature[i].detach().numpy().tolist())
                    db_per_class_cnt[data[0][i].item()] += 1

            self.validate(epoch, save_labels, save_features, db_per_class_cnt)

    def validate(self, epoch, save_labels, save_features, db_per_class_cnt):
        print("start validate")
        self.model.eval()

        PR_pred, PR_true = [], []
        for data in self.val_loader:
            out, feature, cor = self.model(data)
            labels = data[0]
            for i in range(int(feature.shape[0])):
                retrieval_label = retrieval(feature[i].detach().numpy(), labels[i], save_features, save_labels)
                PR_pred.append(retrieval_label)
                PR_true.append(int(labels[i]))

        RtopK, PtopK, FtopK, NDCGtopK, mAP = calcu_evaluation_standard(PR_pred, PR_true, 30, db_per_class_cnt, TopK=100)
        print("R: {}, P: {}, F: {}, NDCG: {}, mAP: {}\n".format(RtopK, PtopK, FtopK, NDCGtopK, mAP))

        self.model.train()


def save_h5(epoch, filenames, features, labels):
    store = pd.HDFStore('./epoch_{}.h5'.format(epoch), 'w')
    store.put('feature', pd.Series(features))
    store.put('label', pd.Series(labels))
    store.put('filename', pd.Series(filenames))
    store.close()


def retrieval(model_feature, y_true, features, labels, topK=100):
    dist = []
    retrieval_list, retrieval_labels, retrieval_features = [], [], []

    for feature in features:
        dist.append(np.linalg.norm(model_feature - feature))

    max_number = heapq.nsmallest(topK, dist)

    max_index = []
    for t in max_number:
        index = dist.index(t)
        retrieval_features.append(features[index])
        max_index.append(index)
        dist[index] = -1
    index = list(max_index)

    for i in index:
        retrieval_labels.append(labels[i])

    return retrieval_labels


def calcu_evaluation_standard(retrieval_labels, y_trues, class_nums, db_per_class_cnt, TopK=100):
    PR_recall = np.zeros((class_nums, TopK))
    PR_precision = np.zeros((class_nums, TopK))
    class_cnt = np.zeros(class_nums)
    DCG, IDCG, mAP = 0, 0, 0

    for idx in range(len(y_trues)):
        per_DCG, per_IDCG, per_mAP, true_cnt = 0, 0, 0, 0
        precisions = np.zeros(TopK)
        recalls = np.zeros(TopK)

        for i in range(TopK):
            if retrieval_labels[idx][i] == y_trues[idx]:
                true_cnt += 1
                per_mAP += true_cnt / (i + 1)
                per_DCG += 1 / log2(i + 2)
            precisions[i] = true_cnt / (i + 1)
            recalls[i] = true_cnt / db_per_class_cnt[y_trues[idx]]

        DCG += per_DCG
        num = int(retrieval_labels[idx].count(y_trues[idx]))
        if num > TopK:
            num = TopK
        for i in range(num):
            per_IDCG += 1 / log2(i + 2)
        IDCG += per_IDCG

        mAP += per_mAP / db_per_class_cnt[y_trues[idx]]

        class_cnt[y_trues[idx]] += 1
        PR_recall[y_trues[idx]] += recalls
        PR_precision[y_trues[idx]] += precisions
    for i in range(class_nums):
        for j in range(TopK):
            PR_recall[i][j] /= class_cnt[i]
            PR_precision[i][j] /= class_cnt[i]

    end_recall = np.zeros(TopK)
    end_precision = np.zeros(TopK)
    for i in range(TopK):
        sum_recall, sum_precision = 0, 0
        for j in range(class_nums):
            sum_recall += PR_recall[j][i]
            sum_precision += PR_precision[j][i]
        end_recall[i] = sum_recall / class_nums
        end_precision[i] = sum_precision / class_nums

    return end_recall[TopK - 1], end_precision[TopK - 1], \
           (2 * end_precision[TopK - 1] * end_recall[TopK - 1]) / (end_recall[TopK - 1] + end_precision[TopK - 1]), \
           DCG / IDCG, mAP / len(y_trues)
