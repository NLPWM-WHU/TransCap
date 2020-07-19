import random

def balance():
    review = open(r'train/review.txt', 'r', encoding='utf-8').readlines()
    label = open(r'train/label.txt', 'r', encoding='utf-8').readlines()
    position = open(r'train/position.txt', 'r', encoding='utf-8').readlines()
    term = open(r'train/term.txt', 'r', encoding='utf-8').readlines()

    review_balanced = open(r'train/balanced_review.txt', 'w', encoding='utf-8')
    label_balanced = open(r'train/balanced_label.txt', 'w', encoding='utf-8')
    position_balanced = open(r'train/balanced_position.txt', 'w', encoding='utf-8')
    term_balanced = open(r'train/balanced_term.txt', 'w', encoding='utf-8')

    pos_list = []
    neu_list = []
    neg_list = []

    lines = label

    for idx, line in enumerate(lines):
        senti = line.strip()
        if senti == 'positive':
            pos_list.append(idx)
        elif senti == 'neutral':
            neu_list.append(idx)
        elif senti == 'negative':
            neg_list.append(idx)

    print(len(pos_list))
    print(len(neu_list))
    print(len(neg_list))
    for i, sample in enumerate(pos_list):
        review_balanced.write(review[pos_list[i]])
        label_balanced.write(label[pos_list[i]])
        position_balanced.write(position[pos_list[i]])
        term_balanced.write(term[pos_list[i]])

        if i < len(neu_list):
            index1 = i
        else :
            index1 = random.randint(0,len(neu_list)-1)
        review_balanced.write(review[neu_list[index1]])
        label_balanced.write(label[neu_list[index1]])
        position_balanced.write(position[neu_list[index1]])
        term_balanced.write(term[neu_list[index1]])

        if i < len(neg_list):
            index2 = i
        else:
            index2 = random.randint(0,len(neg_list)-1)
        review_balanced.write(review[neg_list[index2]])
        label_balanced.write(label[neg_list[index2]])
        position_balanced.write(position[neg_list[index2]])
        term_balanced.write(term[neg_list[index2]])

balance()
