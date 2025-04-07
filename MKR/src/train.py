import tensorflow as tf
import numpy as np
from model import MKR
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    kg = data[7]

    model = MKR(args, n_user, n_item, n_entity, n_relation)

    # top-K evaluation settings
    user_num = 100
    k_list = [10, 20, 50, 100]
    train_record = get_user_record(train_data, True)
    test_record = get_user_record(test_data, False)
    user_list = list(set(train_record.keys()) & set(test_record.keys()))
    if len(user_list) > user_num:
        user_list = np.random.choice(user_list, size=user_num, replace=False)
    item_set = set(list(range(n_item)))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epochs):
            # RS training
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train_rs(sess, get_feed_dict_for_rs(model, train_data, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(loss)

            # KGE training
            if step % args.kge_interval == 0:
                np.random.shuffle(kg)
                start = 0
                while start < kg.shape[0]:
                    _, rmse = model.train_kge(sess, get_feed_dict_for_kge(model, kg, start, start + args.batch_size))
                    start += args.batch_size
                    if show_loss:
                        print(rmse)

            # CTR evaluation
            train_auc, train_acc = model.eval(sess, get_feed_dict_for_rs(model, train_data, 0, train_data.shape[0]))
            eval_auc, eval_acc = model.eval(sess, get_feed_dict_for_rs(model, eval_data, 0, eval_data.shape[0]))
            test_auc, test_acc = model.eval(sess, get_feed_dict_for_rs(model, test_data, 0, test_data.shape[0]))

            print('epoch %d    train auc: %.4f  acc: %.4f    eval auc: %.4f  acc: %.4f    test auc: %.4f  acc: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))

            # top-K evaluation
            if show_topk:
                precision, recall, f1, ndcg = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print()
                print('f1: ', end='')
                for i in f1:
                    print('%.4f\t' % i, end='')
                print()
                print('ndcg: ', end='')
                for i in ndcg:
                    print('%.4f\t' % i, end='')
                print('\n')


def get_feed_dict_for_rs(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2],
                 model.head_indices: data[start:end, 1]}
    return feed_dict


def get_feed_dict_for_kge(model, kg, start, end):
    feed_dict = {model.item_indices: kg[start:end, 0],
                 model.head_indices: kg[start:end, 0],
                 model.relation_indices: kg[start:end, 1],
                 model.tail_indices: kg[start:end, 2]}
    return feed_dict


import numpy as np

def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    f1_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    for user in user_list:
        test_items = list(item_set - train_record[user])
        item_score_map = {}

        items, scores = model.get_scores(sess, {
            model.user_indices: [user] * len(test_items),
            model.item_indices: test_items,
            model.head_indices: test_items
        })

        for item, score in zip(items, scores):
            item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [x[0] for x in item_score_pair_sorted]

        for k in k_list:
            topk_items = item_sorted[:k]
            hits = set(topk_items) & test_record[user]

            precision = len(hits) / k
            recall = len(hits) / len(test_record[user])
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            dcg = 0.0
            idcg = 0.0
            for idx, item in enumerate(topk_items):
                if item in test_record[user]:
                    dcg += 1 / np.log2(idx + 2)
            for idx in range(min(len(test_record[user]), k)):
                idcg += 1 / np.log2(idx + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0

            precision_list[k].append(precision)
            recall_list[k].append(recall)
            f1_list[k].append(f1)
            ndcg_list[k].append(ndcg)

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    f1 = [np.mean(f1_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, f1, ndcg



def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict
