from sklearn.datasets import fetch_mldata
mnemo_minst=fetch_mldata('MNIST original')
mnemo_minst

x,y=mnemo_minst['data'],mnemo_minst['target']
x.shape

y.shape

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
numero=x[23000]

numero

numero_imagen=numero.reshape(28,28)
numero_imagen

plt.imshow(numero_imagen, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()

y[23000]

x_train,x_test,y_train,y_test=x[:60000],x[60000:],y[:60000],y[60000:]

y_train_3=(y_train==3)
y_test_3=(y_test==3)

x_train.shape

y_train_3.shape

y_train_3

from sklearn.linear_model import SGDClassifier
sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(x_train,y_train_3)

y_test.shape

y[23001]

numero_imagen_otro=x[23001].reshape(28,28)
numero_imagen_otro
plt.imshow(numero_imagen_otro, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()

sgd_clf.predict([x[23001]])
sgd_clf.predict([x_test[5000]])

numero_imagen_otro=x_test[5000].reshape(28,28)
numero_imagen_otro
plt.imshow(numero_imagen_otro, cmap=matplotlib.cm.binary, interpolation='nearest')
plt.axis('off')
plt.show()

y_test[5000]

from sklearn.cross_validation import cross_val_score
validacion_cruzada=cross_val_score(sgd_clf,x_train,y_train_3,cv=3,scoring="accuracy")

validacion_cruzada


import numpy as np
index_random=np.random.permutation(60000)
x_train,y_train=x_train[index_random],y_train[index_random]
sgd_clf=SGDClassifier(random_state=42)
y_train_3=(y_train==3)
y_test=(y_test==3)
sgd_clf.fit(x_train,y_train_3)
validacion_cruzada_indexR=cross_val_score(sgd_clf,x_train,y_train_3,cv=3,scoring='accuracy')

validacion_cruzada_indexR

sgd_clf.predict(x_test)


from sklearn.metrics import confusion_matrix
prediccion=sgd_clf.predict(x_test)
confusion_matrix(y_test_3,prediccion)


correctos=(8715+883)/10000
correctos

TP=883
FP=275
FN=127
precision=TP/(TP+FP)
recall=TP/(TP+FN)
valores=(precision,recall)
valores

from sklearn.metrics import f1_score
F1=f1_score(y_test_3,prediccion)
F1