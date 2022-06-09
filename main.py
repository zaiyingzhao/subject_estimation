import MeCab
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import optimizers

#csv読み込み
df = pd.read_csv("./examination_test.csv")
v = df.values
df2 = pd.read_csv("./examination_answer.csv")
v2 = df2.values.tolist()

#分かち書き
wakati = MeCab.Tagger('-Owakati')
for i in range(len(v)):
    v[i][0] = wakati.parse(v[i][0]).split()

#doc2vec
def labelize(trainlabel,s):
    if s == '日本史': return np.append(trainlabel,0)
    elif s == '世界史': return np.append(trainlabel,1)
    elif s == '地理': return np.append(trainlabel,2)
    else: return np.append(trainlabel,3)

trainarray = np.array([])
trainlabel = np.array([])

for i in range(len(v)):
    training_docs = []
    training_docs.append(TaggedDocument(words=v[i][0],tags=v2[i]))
    model = Doc2Vec(documents=training_docs, min_count=1, dm=1, vector_size = 100, sample=10, window = 10)
    if i==0: 
        trainarray = np.concatenate([trainarray, model.docvecs[v2[i][0]]])
        trainlabel = labelize(trainlabel, v2[i][0]) 
        continue
    trainarray = np.vstack([trainarray, model.docvecs[v2[i]][0]])
    trainlabel = labelize(trainlabel, v2[i][0])

trainarray = (trainarray+1)/2.0 #値を0~1へ正規化

#ここからデータの学習
class_names = ['日本史', '世界史', '地理', '公民']

model_study = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4, activation='softmax')
])

opt = keras.optimizers.RMSprop(learning_rate=1e-6, decay=0.0)
model_study.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_study.fit(trainarray, trainlabel, epochs=10000)

#テスト
test_loss, test_acc = model_study.evaluate(trainarray, trainlabel, verbose=2)
print(test_acc)

predictions = model_study.predict(trainarray)

#データ出力
pred = []
for i in range(len(v)):
    pred.append(class_names[np.argmax(predictions[i])])
predarray = np.array(pred)
df_pred = pd.DataFrame(predarray)
df_pred.to_csv("pred.csv", index=False, header=['subject'])