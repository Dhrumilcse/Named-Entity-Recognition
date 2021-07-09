# Imports
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
np.random.seed(0)

# Data Locading and cleaning
data = pd.read_csv("data.csv", encoding="latin1")
data = data.fillna(method="ffill")
words = list(set(data["Word"].values))
words.append("ENDPAD")
tags = list(set(data["Tag"].values))
num_words, num_tags = len(words), len(tags)

# Forms Sentence in form of [[(token1)(pos1)(tag1),(token2),(pos2),(tag2)],....,n]
class SentenceGetter(object):
    """
    Generate Sentence from words in format [(token1,pos1,tag1),....n]
    """
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [
            (w, p, t)
            for w, p, t in zip(
                s["Word"].values.tolist(),
                s["POS"].values.tolist(),
                s["Tag"].values.tolist(),
            )
        ]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

# Get sentences
getter = SentenceGetter(data)
sentences = getter.sentences

# Word and tag indexing, returns following:
    # word2idx = {
    #     'obama': 1,
    #     'was': 5,
    #     'president': 8,
    # }
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

max_len = 50

# Padding words and tags to make them same length
X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words - 1)
print("-"*20)
y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])



# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model construction and training
# Reference - https://arxiv.org/pdf/1508.01991v1.pdf
input_word = Input(shape=(max_len,))
model = Embedding(input_dim=num_words, output_dim=50, input_length=max_len)(input_word)
model = SpatialDropout1D(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(
    model
)
out = TimeDistributed(Dense(num_tags, activation="softmax"))(model)
model = Model(input_word, out)
model.summary()

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Callbacks defined for checkpoint and early stopping
chkpt = ModelCheckpoint(
    "model_weights.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=1,
    verbose=0,
    mode="max",
    baseline=None,
    restore_best_weights=False,
)

callbacks = [chkpt, early_stopping]

# Fit the model
history = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=7,
    callbacks=callbacks,
    verbose=1,
)

# Evaluate the model
print("Evaluate on test data")
results = model.evaluate(x_test, y_test)
print("test loss, test acc:", results)

# Save model to YAML
model.save("ner_model.h5")

# Saving Vocab
with open('word_to_index.pickle', 'wb') as handle:
    pickle.dump(word2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
# Saving Vocab
with open('tag_to_index.pickle', 'wb') as handle:
    pickle.dump(tag2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Results
i = np.random.randint(0, x_test.shape[0])  # 659
p = model.predict(np.array([x_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" * 30)
for w, true, pred in zip(x_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w - 1], tags[true], tags[pred]))

