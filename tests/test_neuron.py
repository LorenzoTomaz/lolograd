import numpy as np
from sklearn.datasets import load_iris, load_digits
from app.engine import Value
from app.examples.conv_mlp import ConvMLP
from app.examples.mlp import MLP
from app.layers.dense import DenseLayer
from app.layers.neuron import Neuron
from app.loss.cross_entropy import CrossEntropyLoss
from app.utils.hot_encode import one_hot_encode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

DIGITS_DATASET = load_digits()

IRIS_DATASET = load_iris()


def test_neuron_fires():
    n = Neuron(2)
    x = [Value(1), Value(2)]
    y = n(x)
    assert abs(y.data) > 0
    assert y.grad == 0
    y.backward()
    assert y.grad != 0
    for w in n.weights:
        assert w.grad != 0
    assert n.bias.grad != 0


def test_layer_activation():
    l = DenseLayer(2, 3)
    x = [Value(1), Value(2)]
    y = l(x)
    assert len(y) == 3
    for y_ in y:
        assert abs(y_.data) > 0
        assert y_.grad == 0
    [y_.backward() for y_ in y]
    for y_ in y:
        assert y_.grad != 0
    for n in l.neurons:
        for w in n.weights:
            assert w.grad != 0
        assert n.bias.grad != 0


def test_mlp_activation():
    mlp = MLP(3, [3, 4, 1])
    x = [Value(1), Value(2), Value(3)]
    y = mlp(x)
    assert len(y) == 1
    for y_ in y:
        assert abs(y_.data) > 0
        assert y_.grad == 0
    [y_.backward() for y_ in y]
    for y_ in y:
        assert y_.grad != 0
    for layer in mlp.layers:
        for n in layer.neurons:
            for w in n.weights:
                assert w.grad != 0
            assert n.bias.grad != 0


def load_iris_dataset_classification():
    X = IRIS_DATASET.data
    y = IRIS_DATASET.target
    return X, y


def test_iris_dataset_classification():
    dataset = load_iris_dataset_classification()
    X, y = dataset
    num_classes = len(set(y))
    y = list(map(lambda x: one_hot_encode(x, num_classes), y))
    mlp_hidden_dim = 50
    model = MLP(X.shape[1], [mlp_hidden_dim, num_classes])
    learning_rate = 0.01
    loss_function = CrossEntropyLoss()
    for epoch in range(10):
        total_loss = 0
        for i in range(0, len(X)):
            x = X[i]
            y_true = y[i]
            y_pred = model(x)
            loss = loss_function(y_pred, y_true)
            total_loss += loss.data
            for p in model.parameters():
                p.grad = 0
            loss.backward()
            for param in model.parameters():
                param.data -= learning_rate * param.grad
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")


def prepare_data(X, y, num_classes):
    X_prepared = [[[Value(pixel) for pixel in row] for row in image] for image in X]
    y_prepared = [one_hot_encode(label, num_classes=num_classes) for label in y]
    return X_prepared, y_prepared


def test_digits_dataset_classification():
    X, y = DIGITS_DATASET.images, DIGITS_DATASET.target

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = X.reshape(X.shape[0], -1)
    X_scaled = scaler.fit_transform(X_scaled)
    X_scaled = X_scaled.reshape(X.shape)
    num_classes = 10  # Digits 0-9
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    X_train_prepared, y_train_prepared = prepare_data(X_train, y_train, num_classes)
    X_test_prepared, y_test_prepared = prepare_data(X_test, y_test, num_classes)
    input_depth = 1  # Grayscale images
    mlp_hidden_dim = 64
    conv_num_filters = 8

    conv_filter_size = 3

    model = ConvMLP(
        input_depth, num_classes, mlp_hidden_dim, conv_num_filters, conv_filter_size
    )
    learning_rate = 0.01
    loss_function = CrossEntropyLoss()
    for epoch in range(2):
        total_loss = 0
        for i in range(0, len(X_train_prepared)):
            x = X_train_prepared[i]
            y_true = y_train_prepared[i]
            y_pred = model(x)
            loss = loss_function(y_pred, y_true)
            total_loss += loss.data
            for p in model.parameters():
                p.grad = 0
            loss.backward()
            for param in model.parameters():
                param.data -= learning_rate * param.grad
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
    test_predictions = []
    for x in X_test_prepared:
        logits = model(x)
        predicted_class = np.argmax([logit.data for logit in logits])
        test_predictions.append(predicted_class)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions))
