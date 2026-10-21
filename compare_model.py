import numpy as np
import classification_0_1 as model1
import classification_all as model2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print("=== So sánh logisticregression ===")
X_train1, y_train1 = model1.load_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
X_test1, y_test1 = model1.load_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

my_model1 = model1.logistic_regression()
my_model1.fit(X_train1, y_train1)
my_preds1 = my_model1.predict_result(X_test1)
my_acc1 = np.mean(my_preds1 == y_test1) * 100

sk_model1 = LogisticRegression(max_iter=300, solver='lbfgs')
sk_model1.fit(X_train1, y_train1)
sk_preds1 = sk_model1.predict(X_test1)
sk_acc1 = accuracy_score(y_test1, sk_preds1) * 100

print(f"Độ chính xác (Model tự viết): {my_acc1:.2f}%")
print(f"Độ chính xác (Scikit-Learn):  {sk_acc1:.2f}%")

print("=== So sánh softmaxregression ===")
X_train2, y_train2 = model2.load_data('train-images.idx3-ubyte', 'train-labels.idx1-ubyte')
X_test2, y_test2 = model2.load_data('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')

my_model2 = model2.softmax_regression(learning_rate=0.5, epochs=300)
my_model2.fit(X_train2, y_train2)
my_preds2 = my_model2.predict_result(X_test2)
my_acc2 = np.mean(my_preds2 == y_test2) * 100

sk_model2 = LogisticRegression(max_iter=300, solver='lbfgs')
sk_model2.fit(X_train2, y_train2)
sk_preds2 = sk_model2.predict(X_test2)
sk_acc2 = accuracy_score(y_test2, sk_preds2) * 100

print(f"\nĐộ chính xác 10 nhãn (my Model): {my_acc2:.2f}%")
print(f"Độ chính xác 10 nhãn (Scikit-Learn):  {sk_acc2:.2f}%")

print("\nBáo cáo chi tiết Sklearn :")
print(classification_report(y_test2, sk_preds2, digits=4))