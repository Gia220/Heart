
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score
import pandas as pd
import os

print("\n--- Addestramento SVM (Kernel RBF) ---")

# 1. Recuperiamo i dati Numpy corrispondenti agli esatti indici di train e test di PyTorch
X_train_svm = X_poly_tensor[train_dataset.indices].numpy()
Y_train_svm = Y_raw[train_dataset.indices].numpy().ravel() # ravel() serve a renderlo 1D per sklearn

X_test_svm = X_poly_tensor[test_dataset.indices].numpy()
Y_test_svm = Y_raw[test_dataset.indices].numpy().ravel()

# 2. Addestriamo la SVM non lineare[cite: 8]
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_svm, Y_train_svm)

# 3. Valutazione
svm_preds = svm_model.predict(X_test_svm)
acc_svm = accuracy_score(Y_test_svm, svm_preds)
f1_svm = f1_score(Y_test_svm, svm_preds, zero_division=0)

print(f"SVM Accuracy: {acc_svm:.4f} | F1-Score: {f1_svm:.4f}")

# 4. Salviamo i risultati nel CSV per il confronto
new_data = pd.DataFrame({
    "Model": ["SVM_RBF_Poly2"],
    "Accuracy": [acc_svm],
    "Precision": [precision_score(Y_test_svm, svm_preds, zero_division=0)],
    "Recall": [recall_score(Y_test_svm, svm_preds, zero_division=0)],
    "F1_Score": [f1_svm]
})
new_data.to_csv("results.csv", mode='a', header=False, index=False)

