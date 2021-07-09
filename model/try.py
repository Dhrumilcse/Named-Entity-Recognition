sentences = [['obama','was','president'],['he','was','in','office','until','2016']]

word2idx = {
    'obama': 1,
    'was': 5,
    'president': 8,
    'he': 4,
    'was': 5,
    'in':90,
    'office': 49,
    'until': 32,
    '2016': 23
}

X_test = []
for s in range(len(sentences)):
    temp = []
    for w in range(len(sentences[s])):
        temp.append(word2idx[sentences[s][w]])
    X_test.append(temp)

print(X_test)