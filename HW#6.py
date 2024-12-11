import pandas as pd
import matplotlib as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

sep = '-'*80
path = '/Users/cadepreister/Desktop/Intro to Python Data Analytics/'

def one_valid():
    digits = datasets.load_digits()
    digits_data = digits.data
    digits_targets = digits.target

    results = []
    runs = 0

    while True:
        print(f"{sep}\nPick your specifications for run {runs+1}!")
        layers = input("Enter how many layers? (1 - 4), or q to quit: ").strip()
        if layers.lower() == 'q':
            break

        try:
            layers = int(layers)
            if 1 > int(layers) or int(layers) > 4:
                raise ValueError("Number must be between 1 & 4!")

            neurons = input("Enter how many neurons?\n"
                            "Example (3 layers): X, X, X (X = integer between 10 & 100): ").strip()
            neurons = list(map(int, neurons.split(',')))
            if len(neurons) != layers or any(n < 10 or n > 100 for n in neurons):
                raise ValueError("Check your layers & formatting!"
                                 "\nExample (3 layers): X, X, X (X = integer between 10 & 100):")

            iterations = input("Enter how many iterations? (200 - 1000): ").strip()
            iterations = int(iterations)
            if 200 > int(iterations) or int(iterations) > 1000:
                raise ValueError("Number must be between 200 & 1000!")

            activation = input("Enter activation function ('relu', 'tanh', 'logistic', 'identity'): ").strip()
            if activation.strip().lower() not in ['relu', 'tanh', 'logistic', 'identity']:
                raise ValueError("Activation must be 'relu', 'tanh', 'logistic', or 'identity'.")

        except ValueError as e:
            print(f"Error: {e}")

        else:
            runs += 1
            one_p1(digits_data, digits_targets, layers, neurons, iterations, activation, runs, results)

    if results:
        return results

def one_p1(data, targets, layers, neurons, iterations, activation, runs, results):

    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    hls = neurons
    a = 0.0001

    clf = MLPClassifier(hidden_layer_sizes=hls, max_iter=iterations, alpha=a, activation=activation, random_state=42)
    clf.fit(x_train, y_train)

    y_predict = clf.predict(x_test)

    accuracy = accuracy_score(y_test, y_predict)
    confusion_mtrx = confusion_matrix(y_test, y_predict)

    print(f"{sep}\nNumber of rows and columns of the dataset: {data.shape}")

    print(clf)

    print("Test sample labels:")
    print(y_test)
    print("Test samples classified as:")
    print(y_predict)

    print("Confusion Matrix:")
    print(confusion_mtrx)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Training set score: {clf.score(x_train, y_train):.2f}")
    print(f"Test set score: {clf.score(x_test, y_test):.2f}\n{sep}")

    for idx, weights in enumerate(clf.coefs_):
        print(f"Weights between layer {idx} and layer {idx + 1}:")
        print(weights)

    results.append({
        "Runs":runs,
        "Layers":layers,
        "Neurons":neurons,
        "Accuracy":accuracy,
        "Alpha":a,
        "Activation":activation,
        "Random State":42
    })

def create_file(results):
    usr_file = input("What would you like the name of your file to be (new_dataset.xlsx): ")
    if not usr_file.endswith('.xlsx'):
        usr_file += '.xlsx'

    df = pd.DataFrame(results)
    df.to_excel(path + f'{usr_file}.xlsx', index=False)
    print(f"Data Saved!!! To -> {usr_file}")

def main():
    final_results = one_valid()
    create_file(final_results)

if __name__ == "__main__":
    main()