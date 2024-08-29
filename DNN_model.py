import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from matplotlib.backends.backend_pdf import PdfPages

def preprocess_data(file_path, columns_to_drop, nrows=None):
    df = pd.read_csv(file_path, nrows=nrows)
    df = df.drop(columns_to_drop, axis=1)
    X = df.drop('pile_up', axis=1)
    y = df['pile_up']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, X.columns

columns_to_drop = ['cluster_SECOND_TIME', 'clusterPhi', 'cluster_ENG_CALIB_TOT', 'cluster_nCells',
                   'cluster_sumCellE', 'cluster_ENG_CALIB_FRAC_EM', 'nCluster', 'cluster_CENTER_Z',
                   'cluster_CENTER_X', 'cluster_CENTER_Y']

X_train, y_train, features_train = preprocess_data('transformed_data_quantile_trn_cut.csv', columns_to_drop, nrows=500000)
X_val, y_val, _ = preprocess_data('transformed_data_quantile_val_cut.csv', columns_to_drop, nrows=500000)
X_test, y_test, features_test = preprocess_data('transformed_data_quantile_tst_cut.csv', columns_to_drop, nrows=500000)

model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.2),
    Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
    Dropout(0.1),
    Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), verbose=1)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

avg_accuracy = np.mean(history.history['accuracy'])
avg_val_accuracy = np.mean(history.history['val_accuracy'])
avg_loss = np.mean(history.history['loss'])
avg_val_loss = np.mean(history.history['val_loss'])

print(f"Average Accuracy: {avg_accuracy:.4f}")
print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Average Validation Loss: {avg_val_loss:.4f}")

print('done training')

exit()

#save the model
model.save('/data1/ihuang/keras_model.keras')

def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/data1/ihuang/training_validation.png')

plot_metrics(history)

def create_histograms(X, y, features, file_path):
    pdf_pages = PdfPages(file_path)
    positive_mask = y == 1
    negative_mask = y == 0
    num_bins = 30

    for feature in features:
        feature_index = features.get_loc(feature)
        positive_data = X[positive_mask, feature_index]
        negative_data = X[negative_mask, feature_index]

        plt.figure(figsize=(10, 6))
        plt.hist(positive_data, bins=num_bins, alpha=0.5, label='Positive Class (Pile Up)', density=True, color='blue')
        plt.hist(negative_data, bins=num_bins, alpha=0.5, label='Negative Class (No Pile Up)', density=True, color='red')
        plt.title(f'Histogram of {feature}')
        plt.xlabel('Feature Value')
        plt.ylabel('Normalized Frequency')
        plt.legend()
        pdf_pages.savefig()
        plt.close()

    pdf_pages.close()

create_histograms(X_test, y_test, features_test, '/data1/ihuang/feature_histograms_test_data.pdf')

def calculate_feature_importance(model, X, y):
    original_accuracy = model.evaluate(X, y, verbose=0)[1]
    importances = []

    for i in range(X.shape[1]):
        X_shuffled = np.copy(X)
        np.random.shuffle(X_shuffled[:, i])
        shuffled_accuracy = model.evaluate(X_shuffled, y, verbose=0)[1]
        importance = original_accuracy - shuffled_accuracy
        importances.append(importance)

    return np.array(importances)

importances = calculate_feature_importance(model, X_test, y_test)

plt.figure(figsize=(12, 6))
feature_indices = np.argsort(importances)[::-1]
plt.bar(range(X_test.shape[1]), importances[feature_indices], align='center')
plt.xticks(range(X_test.shape[1]), features_test[feature_indices], rotation=90)
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('/data1/ihuang/feature_importances.png')
plt.close()

exit()

y_pred_prob = model.predict(X_test, verbose=0)
prob_df = pd.DataFrame(y_pred_prob, columns=['Probability of Pileup'])
prob_df['Probability of No Pileup'] = 1 - prob_df['Probability of Pileup']

plt.figure(figsize=(10, 6))
plt.hist(prob_df['Probability of No Pileup'], bins=30, alpha=0.7, color='blue', label='Probability of No Pileup')
plt.hist(prob_df['Probability of Pileup'], bins=30, alpha=0.7, color='red', label='Probability of Pileup')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Predicted Probability Histograms')
plt.legend()
plt.savefig('/data1/ihuang/predicted_probability_histograms.pdf')

y_test_np = y_test.to_numpy()
pileup = y_pred_prob[y_test_np == 1]
nopileup = y_pred_prob[y_test_np == 0]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].hist(pileup, bins=50, range=(0, 1), histtype='step', color='red', label='Pileup')
ax[0].hist(1 - pileup, bins=50, range=(0, 1), histtype='step', color='blue', label='No Pileup')
ax[0].set(title='Predicted Probability (Pileup)', xlabel='Probability', ylabel='Frequency')

ax[1].hist(nopileup, bins=50, range=(0, 1), histtype='step', color='red', label='Pileup')
ax[1].hist(1 - nopileup, bins=50, range=(0, 1), histtype='step', color='blue', label='No Pileup')
ax[1].set(title='Predicted Probability (No Pileup)', xlabel='Probability', ylabel='Frequency')

plt.legend()
plt.tight_layout()
plt.savefig('/data1/ihuang/pp.png')

print("All plots have been saved in /data1/ihuang/")
