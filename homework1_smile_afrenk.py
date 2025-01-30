import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
def fPC (y, yhat):
    return np.mean(y == yhat)
def measureAccuracyOfPredictors (predictors, X, y):
    n = X.shape[0]
    m = len(predictors)

    preds = np.zeros((n, m))
    for j, (r1, c1, r2, c2) in enumerate(predictors):
        preds[:, j] = (X[:, r1, c1] > X[:, r2, c2]).astype(int)

    yhat = (np.mean(preds, axis=1) > 0.5).astype(int)
    return(fPC(y, yhat))

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces,
testingLabels, feats=6):
    n, img_size, _ = trainingFaces.shape  # assumes image of shape n by 24 px by 24 px as per line 73

    feat = []

    for j in range(feats):
        best_acc = 0
        best_feat = None
        r1g, c1g, r2g, c2g = np.meshgrid(np.arange(img_size), np.arange(img_size), np.arange(img_size), np.arange(img_size), indexing='ij')
        # need to flatten grids that I made
        r1g = r1g.ravel()
        c1g = c1g.ravel()
        r2g = r2g.ravel()
        c2g = c2g.ravel()

        #  mask for valid indices
        valid_idx_mask  = (r1g != r2g) | (c1g != c2g)
        r1g, c1g, r2g, c2g = r1g[valid_idx_mask], c1g[valid_idx_mask], r2g[valid_idx_mask], c2g[valid_idx_mask]
        feature_matrix = (trainingFaces[:, r1g, c1g] > trainingFaces[:, r2g, c2g]).astype(int)
        for idx in range(len(r1g)):
            feature_candidate = (r1g[idx], c1g[idx], r2g[idx], c2g[idx])
            accuracy = measureAccuracyOfPredictors(feat + [feature_candidate], trainingFaces, trainingLabels)
            if accuracy > best_acc:
                best_acc = accuracy
                best_feat = feature_candidate

        feat.append(best_feat)
        print(f'Round{j+1} picked {best_feat} that had accuracy {best_acc:.4f}')

    train_acc = measureAccuracyOfPredictors(feat, trainingFaces, trainingLabels)
    test_acc = measureAccuracyOfPredictors(feat, testingFaces, testingLabels)

    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Testing Accuracy: {test_acc:.4f}")
    return feat, train_acc, test_acc


def accforpdf(trainingFaces, trainingLabels, testingFaces, testingLabels):
    training_sizes = range(400, 2001, 200)
    results = []
    for n in training_sizes:
        subset_faces = trainingFaces[:n]
        subset_labels = trainingLabels[:n]

        selected_features, train_acc, test_acc = stepwiseRegression(subset_faces, subset_labels, testingFaces, testingLabels, feats=6)


        results.append((n, train_acc, test_acc))
        print(f"n = {n}: Training Accuracy = {train_acc:.4f}, Testing Accuracy = {test_acc:.4f}")


    results_array = np.array(results, dtype=[('n', 'i4'), ('training_accuracy', 'f4'), ('testing_accuracy', 'f4')])
    return results_array


def visualize_learned_features(testingFaces, selected_features):
    # abstracted into separate func so I could play around with mpl syntax

    example_image = testingFaces[0]

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(example_image, cmap='gray')

    for i, (r1, c1, r2, c2) in enumerate(selected_features):
 
        rect1 = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect1)
        ax.text(c1, r1, f'F{i+1}', color='red', fontsize=10)


        rect2 = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect2)


    plt.title("Visualization of Learned Features")
    plt.axis('off')
    plt.show()


def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24) # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    selected_features, train_acc, test_acc = stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels, feats=6)
    visualize_learned_features(testingFaces, selected_features)
    accuracy_results = accforpdf(trainingFaces, trainingLabels, testingFaces, testingLabels)
    plt.figure()
    plt.plot(accuracy_results['n'], accuracy_results['training_accuracy'], label="Training Accuracy")
    plt.plot(accuracy_results['n'], accuracy_results['testing_accuracy'], label="Testing Accuracy")
    plt.xlabel("Number of Training Examples (n)")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Testing Accuracy as a Function of n")
    plt.legend()
    plt.grid(True)
    plt.show()