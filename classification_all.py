import numpy as np
import idx2numpy

def load_data(file_img, file_label):
    images = idx2numpy.convert_from_file(file_img)
    labels = idx2numpy.convert_from_file(file_label)
    X = images.astype(np.float32).reshape(-1, 28*28) / 255.0
    y = labels.astype(np.float32)
    return X, y

X_train, y_train = load_data('train-images.idx3-ubyte','train-labels.idx1-ubyte')
X_test, y_test = load_data('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')

class softmax_regression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None 
        self.bias = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def loss_fn(self, y_one_hot, y_pred):
        esp = 1e-15
        return -np.mean(np.sum(y_one_hot * np.log(y_pred + esp), axis=1))

    def fit(self, X, y, classes=10):
        m, n = X.shape
        y_one_hot = np.eye(classes)[y.astype(int)]
        self.weights = np.zeros((n, classes), dtype=np.float32)
        self.bias = np.zeros((1, classes), dtype=np.float32)
        self.losses = []

        for epoch in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.softmax(linear)

            dw = np.dot(X.T, (y_pred - y_one_hot))/m
            db = np.mean(y_pred - y_one_hot, axis=0, keepdims=True)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (epoch + 1) % 100 == 0 or epoch == 0:
                loss = self.loss_fn(y_one_hot, y_pred,)
                self.losses.append(loss)
                print(f"Epoch {epoch + 1:4d}/{self.epochs} - Loss: {loss:.2f}")

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Call fit() before predict().")
        linear = np.dot(X, self.weights) + self.bias
        return self.softmax(linear)

    def predict_result(self, X, threshold=0.5):
        return np.argmax(self.predict(X), axis=1)

def report(y_true, y_pred, pos_label=1):
    esp = 1e-15
    TP = np.sum((y_true == pos_label) & (y_pred == pos_label))
    TN = np.sum((y_true != pos_label) & (y_pred != pos_label))
    FP = np.sum((y_true != pos_label) & (y_pred == pos_label))
    FN = np.sum((y_true == pos_label) & (y_pred != pos_label))
    precision = TP / (TP + FP + esp)
    recall = TP / (TP + FN + esp)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == '__main__':
    model = softmax_regression(learning_rate=0.5, epochs=300)
    print("Bắt đầu huấn luyện...")
    model.fit(X_train, y_train)

    test_predictions = model.predict_result(X_test)
    test_accuracy = np.mean(test_predictions == y_test) * 100

    print(f"\nĐộ chính xác (trên tập test): {test_accuracy:.2f}%")
    print("==================================================")
    print("Chi tiết Precision - Recall - F1 từng số 0-9:")
    for i in range(10):
        print(f"Nhãn {i}:")
        report(test_predictions,y_test,i)