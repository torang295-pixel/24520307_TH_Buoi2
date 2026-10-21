import numpy as np
import idx2numpy

def load_data(file_img, file_label):
    images = idx2numpy.convert_from_file(file_img)
    labels = idx2numpy.convert_from_file(file_label)
    target = (labels == 0) | (labels == 1)
    selected_images = images[target]
    selected_labels = labels[target]
    X = selected_images.astype(np.float32).reshape(-1, 28*28) / 255.0
    y = selected_labels.astype(np.float32)
    return X, y

X_train, y_train = load_data('train-images.idx3-ubyte','train-labels.idx1-ubyte')
X_test, y_test = load_data('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')

class logistic_regression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss_fn(self, y_true, y_pred):
        esp = 1e-15
        loss = -np.mean(y_true * np.log(y_pred + esp) + (1 - y_true) * np.log(1 - y_pred + esp))
        return loss

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.losses = []

        for epoch in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)
            dw = np.dot(X.T, (y_pred - y)) / m
            db = np.mean(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if (epoch + 1) % 100 == 0 or epoch == 0:
                loss = self.loss_fn(y, y_pred)
                self.losses.append(loss)
                print(f"Epoch {epoch + 1:4d}/{self.epochs} - Loss: {loss:.2f}")

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Call fit() before predict().")
        linear = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear)

    def predict_result(self, X, threshold=0.5):
        probs = self.predict(X)
        return (probs >= threshold).astype(int)

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
    model = logistic_regression()
    print("Đang huấn luyện: ")
    model.fit(X_train, y_train)

    test_predictions = model.predict_result(X_test)
    test_accuracy = np.mean(test_predictions == y_test) * 100
    
    print(f"Độ chính xác (trên tập test): {test_accuracy}")
    print("=====================================")
    print("Precision - Recall - F1 trên nhãn 1:")
    report(y_test,test_predictions,1)
    print("=====================================")
    print("Precision - Recall - F1 trên nhãn 0:")
    report(y_test,test_predictions,0)
