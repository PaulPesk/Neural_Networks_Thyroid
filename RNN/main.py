import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

def print_random_samples_top_5_probabilities(predicted_probabilities, true_labels, encoder, num_samples=5):
    true_labels_decoded = encoder.inverse_transform(np.argmax(true_labels, axis=1))
    class_labels = encoder.inverse_transform(np.arange(predicted_probabilities.shape[1]))
    random_indices = np.random.choice(np.arange(predicted_probabilities.shape[0]), size=num_samples, replace=False)
    data = []
    for i in random_indices:
        probabilities = predicted_probabilities[i]
        top_indices = np.argsort(probabilities)[-5:][::-1]
        top_probabilities = probabilities[top_indices]
        formatted_probabilities = [f"{class_labels[idx]}: {prob * 100:.2f}%" for idx, prob in
                                   zip(top_indices, top_probabilities)]
        data.append(formatted_probabilities + [true_labels_decoded[i]])
    df = pd.DataFrame(data, columns=[f'Predicted {i + 1}' for i in range(5)] + ['True Label'])
    print(df)
    print()

# Load and preprocess data
column_names = ["age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds",
                "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
                "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
                "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4", "T4U_measured",
                "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG", "referral_source", "target", "patient_id"]

data = pd.read_csv('thyroidDF.csv', names=column_names, na_values=['', ' '], header=None, skiprows=1)
data = data.replace({'f': 0, 't': 1})
data.drop(['patient_id', 'referral_source'], axis=1, inplace=True)
data['sex'] = data['sex'].map({'F': 0, 'M': 1})
data['sex'].fillna(0, inplace=True)
data['target'].fillna('missing', inplace=True)

X = data.drop('target', axis=1)
y = data['target']

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
y_one_hot = to_categorical(encoded_Y)

numerical_columns = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
for col in numerical_columns:
    data[col] = data.groupby('target')[col].transform(lambda x: x.fillna(x.mean()))
X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].mean())

scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

num_classes = y_one_hot.shape[1]
num_features = X.shape[1]

# Initialize lists for history, accuracies, losses, and stopping epochs
history_90_10, history_80_20, history_70_30, history_60_40 = [], [], [], []
accuracies, accuracies1, accuracies2, accuracies3 = [], [], [], []
losses, losses1, losses2, losses3 = [], [], [], []
stop_epochs_90_10, stop_epochs_80_20, stop_epochs_70_30, stop_epochs_60_40 = [], [], [], []

early_stopping = EarlyStopping(monitor='accuracy', patience=10, mode='max', min_delta=0.00001,
                               restore_best_weights=True, verbose=0)

trial_num = 1
for x in range(10):
    print(f"########################## Test {trial_num} ##################################")
    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(num_features, 1)),  # LSTM instead of SimpleRNN
        Dense(num_classes, activation='softmax')
    ])
    #model = Sequential([
     #   SimpleRNN(50, activation='relu', input_shape=(num_features, 1)),
       # Dense(num_classes, activation='softmax')
    #])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'mean_squared_error'])

    X_reshaped = X.values.reshape((X.shape[0], num_features, 1))

    # 90-10 split
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_one_hot, test_size=0.1, random_state=42 + x)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=32,
                        callbacks=[early_stopping], verbose=0, shuffle=True)
    history_90_10.append(history.history)
    stop_epochs_90_10.append(early_stopping.stopped_epoch + 1 if early_stopping.stopped_epoch != 0 else 100)
    test_loss, test_acc, _ = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(test_acc)
    losses.append(test_loss)

    # Predicted probabilities for 90-10 split
    predicted_probabilities = model.predict(X_test)
    print("90-10 Split - Top 5 predicted probabilities and true labels for 5 random samples:")
    print_random_samples_top_5_probabilities(predicted_probabilities, y_test, encoder)

    # 80-20 split
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_reshaped, y_one_hot, test_size=0.2, random_state=42 + x)
    history1 = model.fit(X_train1, y_train1, epochs=100, validation_data=(X_test1, y_test1),
                         batch_size=32, callbacks=[early_stopping], verbose=0, shuffle=True)
    history_80_20.append(history1.history)
    stop_epochs_80_20.append(early_stopping.stopped_epoch + 1 if early_stopping.stopped_epoch != 0 else 100)
    test_loss1, test_acc1, _ = model.evaluate(X_test1, y_test1, verbose=0)
    accuracies1.append(test_acc1)
    losses1.append(test_loss1)

    predicted_probabilities1 = model.predict(X_test1)
    print("80-20 Split - Top 5 predicted probabilities and true labels for 5 random samples:")
    print_random_samples_top_5_probabilities(predicted_probabilities1, y_test1, encoder)

    # 70-30 split
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_reshaped, y_one_hot, test_size=0.3, random_state=42 + x)
    history2 = model.fit(X_train2, y_train2, epochs=100, validation_data=(X_test2, y_test2),
                         batch_size=32, callbacks=[early_stopping], verbose=0, shuffle=True)
    history_70_30.append(history2.history)
    stop_epochs_70_30.append(early_stopping.stopped_epoch + 1 if early_stopping.stopped_epoch != 0 else 100)
    test_loss2, test_acc2, _ = model.evaluate(X_test2, y_test2, verbose=0)
    accuracies2.append(test_acc2)
    losses2.append(test_loss2)

    predicted_probabilities2 = model.predict(X_test2)
    print("70-30 Split - Top 5 predicted probabilities and true labels for 5 random samples:")
    print_random_samples_top_5_probabilities(predicted_probabilities2, y_test2, encoder)

    # 60-40 split
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_reshaped, y_one_hot, test_size=0.4, random_state=42 + x)
    history3 = model.fit(X_train3, y_train3, epochs=100, validation_data=(X_test3, y_test3),
                         batch_size=32, callbacks=[early_stopping], verbose=0, shuffle=True)
    history_60_40.append(history3.history)
    stop_epochs_60_40.append(early_stopping.stopped_epoch + 1 if early_stopping.stopped_epoch != 0 else 100)
    test_loss3, test_acc3, _ = model.evaluate(X_test3, y_test3, verbose=0)
    accuracies3.append(test_acc3)
    losses3.append(test_loss3)

    predicted_probabilities3 = model.predict(X_test3)
    print("60-40 Split - Top 5 predicted probabilities and true labels for 5 random samples:")
    print_random_samples_top_5_probabilities(predicted_probabilities3, y_test3, encoder)

    trial_num += 1

# Print out the stopping epochs for each trial
print("Stopping epochs for 90-10 split:", stop_epochs_90_10)
print("Stopping epochs for 80-20 split:", stop_epochs_80_20)
print("Stopping epochs for 70-30 split:", stop_epochs_70_30)
print("Stopping epochs for 60-40 split:", stop_epochs_60_40)

# Summary of accuracies and stopping epochs
print("All Accuracies for 90-10 split:", accuracies)
print("All Losses for 90-10 split:", losses)
print("All Accuracies for 80-20 split:", accuracies1)
print("All Losses for 80-20 split:", losses1)
print("All Accuracies for 70-30 split:", accuracies2)
print("All Losses for 70-30 split:", losses2)
print("All Accuracies for 60-40 split:", accuracies3)
print("All Losses for 60-40 split:", losses3)


best_accuracies = [max(accuracies), max(accuracies1), max(accuracies2), max(accuracies3)]
best_losses = [min(losses), min(losses1), min(losses2), min(losses3)]

best_trial_indices = [accuracies.index(best_accuracies[0]),
                      accuracies1.index(best_accuracies[1]),
                      accuracies2.index(best_accuracies[2]),
                      accuracies3.index(best_accuracies[3])]

best_histories = [history_90_10[best_trial_indices[0]],
                  history_80_20[best_trial_indices[1]],
                  history_70_30[best_trial_indices[2]],
                  history_60_40[best_trial_indices[3]]]

y_axis_limit_accuracy = (0, 1)
y_axis_limit_loss = (0, max(max(losses), max(losses1), max(losses2), max(losses3)))
num_epochs = 100

# Plot accuracy and val_accuracy per epoch for each split in separate figures
for i, history in enumerate(best_histories):
    plt.figure(figsize=(8, 6))
    plt.plot(history['accuracy'], label="Training Accuracy")
    plt.plot(history['val_accuracy'], label="Validation Accuracy")
    plt.title(f"Accuracy per Epoch for Split {90 - 10 * i}-{10 - 10 * i}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xlim(0, num_epochs - 1)
    plt.ylim(*y_axis_limit_accuracy)
    plt.xticks(range(0, num_epochs, 10))
    plt.legend()
    plt.show()

# Plot loss and val_loss per epoch for each split in separate figures
for i, history in enumerate(best_histories):
    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'], label="Training Loss")
    plt.plot(history['val_loss'], label="Validation Loss")
    plt.title(f"Loss per Epoch for Split {90 - 10 * i}-{10 - 10 * i}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(0, num_epochs - 1)
    plt.ylim(*y_axis_limit_loss)
    plt.xticks(range(0, num_epochs, 10))
    plt.legend()
    plt.show()

