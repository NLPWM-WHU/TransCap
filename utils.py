from collections import Counter
import numpy as np
import tensorflow as tf
import spacy
en_nlp = spacy.load("en")


def get_position(sptoks, position):
    from_idx = int(position.split(',')[0])
    to_idx = int(position.split(',')[1])
    if from_idx == to_idx == 0:
        pos_info = [0] * len(sptoks)
    else:
        aspect_is = []
        for sptok in sptoks:
            if sptok.idx < to_idx and sptok.idx + len(sptok.text) > from_idx:
                aspect_is.append(sptok.i)
        pos_info = []
        for _i, sptok in enumerate(sptoks):
            pos_info.append(min([abs(_i - i) for i in aspect_is]) + 1)

    return pos_info


def get_data_label(label):
    lab = None
    if label == 'negative':
        lab = [1, 0, 0]
    elif label == 'neutral':
        lab = [0, 1, 0]
    elif label == "positive":
        lab = [0, 0, 1]
    else:
        raise ValueError("Unknown label: %s" % lab)

    return lab


def data_init(path, DSC):
    source_count = []
    source_word2idx = {}
    max_sent_len = 0
    for process in ['train/review.txt', 'train/{}_review.txt'.format(DSC), 'dev/review.txt', 'test/review.txt']:
        print('Processing {}...'.format(process))
        fname = path + process

        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            source_words = []
            for line in lines:
                sptoks = en_nlp(line.strip())
                source_words.extend([sp.text.lower() for sp in sptoks])
                if len(sptoks) > max_sent_len:
                    max_sent_len = len(sptoks)

        if len(source_count) == 0:
            source_count.append(['<pad>', 0])
        source_count.extend(Counter(source_words).most_common())
        for word, _ in source_count:
            if word not in source_word2idx:
                source_word2idx[word] = len(source_word2idx)

    # print(source_count)
    # print(source_word2idx)
    print('max_sentence_length', max_sent_len)

    with open(path+DSC+'_word2id.txt', 'w', encoding='utf-8') as f:
        f.write(str(source_word2idx))

    return source_word2idx


def read_data(fname, source_word2idx, mode=None):
    source_data, target_data, target_label = list(), list(), list()
    source_loc = list()
    target_mask = list()
    max_length = 85
    target_maxlen = 25
    target_mode = list()

    review = open(fname + r'review.txt', 'r', encoding='utf-8').readlines()
    label = open(fname + r'label.txt', 'r', encoding='utf-8').readlines()
    term = open(fname + r'term.txt', 'r', encoding='utf-8').readlines()
    position = open(fname + r'position.txt', 'r', encoding='utf-8').readlines()
    for index, _ in enumerate(review):
        '''
        Word Index
        '''
        sptoks = en_nlp(review[index].strip())

        idx = []
        mask = []
        len_cnt = 0
        for sptok in sptoks:
            if len_cnt < max_length:
                idx.append(source_word2idx[sptok.text.lower()])
                mask.append(1.)
                len_cnt += 1
            else:
                break

        source_data.append(idx + [0] * (max_length - len(idx)))

        '''
        Position Information
        '''
        if mode == 'ASC':
            pos_info = get_position(sptoks, position[index].strip())
        elif mode == 'DSC':
            pos_info = get_position(sptoks, '0,0')
        source_loc.append(pos_info + [0] * (max_length - len(idx)))

        '''
        Term Index
        '''
        if mode == 'ASC':
            t_sptoks = en_nlp(term[index].strip())
            tar_idx = []
            tar_mask = []
            # print(review[index].strip())
            # print(term[index].strip())
            for t_sptok in t_sptoks:
                tar_idx.append(source_word2idx[t_sptok.text.lower()])
                tar_mask.append(1.)

            target_data.append(tar_idx + [0] * (target_maxlen - len(tar_idx)))
            target_mask.append(tar_mask + [0.] * (target_maxlen - len(tar_idx)))
            target_mode.append([1., 0.])
        elif mode == 'DSC':
            target_data.append([0] * target_maxlen)
            target_mask.append([1.] * target_maxlen)
            target_mode.append([0., 1.])

        senti = get_data_label(label[index].strip())
        target_label.append(senti)

    return np.array(source_data), \
           np.array(target_data), \
           np.array(target_label), \
           np.array(target_mask), \
           np.array(source_loc), \
           np.array(target_mode)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def init_word_embeddings(path, word2idx, DSC):
    print('path', path)
    wt = np.random.normal(0, 0.05, [len(word2idx), 300])
    with open('data/glove.840B.300d.txt', 'r',encoding= 'utf-8') as f:
        for line in f:
            content = line.strip().split()
            if content[0] in word2idx:
                #print(is_number(content[1]))
                if is_number(content[1]) == False: continue
                wt[word2idx[content[0]]] = np.array(list(map(np.float32, content[1:])))
    wt = np.asarray(wt, dtype=np.float32)
    wt[0,:] = 0.0
    np.save(path + DSC + '_word_embedding.npy', wt)
    return wt

def separate_hinge_loss(prediction, label, class_num, mode, gamma):
    '''
    negative -0
    neutral  -1
    positive -2
    m_plus = 0.9
    m_minus = 0.1
    lambda_val = 0.5
    '''
    loss = 0.0
    for category in range(class_num):
        if category == 0: #negative
            m_plus = 0.9
            m_minus = 0.1
            lambda_val = 0.5
        elif category == 1: #neutral
            m_plus = 0.9
            m_minus = 0.1
            lambda_val = 0.5
        elif category == 2: #positive
            m_plus = 0.9
            m_minus = 0.1
            lambda_val = 0.5

        vector = prediction[:,category]
        T_c = label[:,category]

        max_l = tf.square(tf.maximum(0., m_plus - vector))
        max_r = tf.square(tf.maximum(0., vector - m_minus))

        origin_L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r #[batch]
        scale_L_c = origin_L_c * gamma

        L_c_concat = tf.concat([tf.expand_dims(origin_L_c,-1), tf.expand_dims(scale_L_c, -1)], -1) # [b,2]
        L_c = tf.reduce_sum(L_c_concat * mode, -1)

        margin_loss = tf.reduce_mean(L_c) #

        loss += margin_loss

    return loss