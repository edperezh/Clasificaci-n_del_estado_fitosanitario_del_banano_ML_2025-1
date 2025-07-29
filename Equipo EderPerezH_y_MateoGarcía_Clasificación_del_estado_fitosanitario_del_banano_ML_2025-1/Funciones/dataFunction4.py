import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset
import torch.nn as nn





def ver_planta(df, idx):
    #longitud_onda = df.columns[3:].astype(float).values
    y = df.iloc[idx, 3:].values
    dpi = df.iloc[idx, 0]
    sana = df.iloc[idx, 1]
    tratamiento = df.iloc[idx, 2]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes()

    ax.plot(y, marker='o', markersize=3, linestyle=None, color="#00BAA0", label=f'dpi:{dpi}, {tratamiento}: {sana}')

    plt.legend()
    plt.show()


def plot_coefs(coefs, index):
    colores = ["#3E1FC8", "#FF7902", "#47E802"]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3,  figsize=(10, 6))
    axs = (ax0, ax1, ax2)
    for i in range(len(axs)):
        ax = axs[i]
        y = coefs[i, :]
        ax.plot(index, y, color=colores[i],marker='o', markersize=3, linestyle='None',label=r"Longitud onda / Pesos")
        #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.legend()
    plt.show()



def plot_coefs_2(coefs, index):
    colores = ["#3E1FC8", "#FF7902", "#47E802"]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(nrows=3, figsize=(10, 6), sharex=True)
    
    for i, ax in enumerate(axs):
        y = coefs[i, :]
        ax.plot(index, y, color=colores[i], marker='o', markersize=3, linestyle='None', label="Longitud onda / Pesos")
        ax.set_xlim(400, max(index))  # Ajustar el rango en X
        ax.tick_params(axis='both', labelsize=10)
        
        # Ajustar márgenes y eliminar espacios vacíos
        ax.margins(x=0, y=0.1)
    
    # Agregar una leyenda única fuera de la gráfica
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.1, 1), fontsize=12)

    plt.tight_layout()
    plt.show()


def kfoldCV_clf(model, X, y, n_splits):
            kfold = KFold(n_splits=n_splits).split(X, y)
            scores = []
            for k, (train, test) in enumerate(kfold):
                model.fit(X[train], y[train])
                score = model.score(X[test], y[test])
                scores.append(score)
                print(f'fold {k+1}', f'accuracy {score:.3f}')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f'\nCV accuracy: {mean_score:.3f} +/- {std_score:.3f}')



def model_confmat(model, X, y):
    accuracy = model.score(X, y)
    print(f'\naccuracy: {accuracy:.3f}')
    y_predict = model.predict(X)
    confmat = confusion_matrix(y_pred=y_predict, y_true=y)
    print()
    print('Confusion_Matrix:')
    print(confmat)
    clases = np.unique(y)

    plt.style.use('classic')
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes()

    ax.matshow(confmat, cmap='viridis', alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    n_labels = [i for i in range(confmat.shape[0])]
    ax.set_xticks(n_labels)
    ax.set_yticks(n_labels)
    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)

    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Predict label')
    plt.ylabel('True Label')
    plt.show()


def model_confmat_2(y_pred, y_test):
    confmat= confusion_matrix(y_pred=y_pred, y_true=y_test)

    accuracy = confmat[0, 0] + confmat[1, 1] + confmat[2, 2]
    accuracy = accuracy/y_test.shape[0]
    print(f'accuracy: {accuracy*100:.2f}')

    plt.style.use('classic')
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes()

    ax.matshow(confmat, cmap='viridis', alpha=0.3)

    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    n_labels = [i for i in range(confmat.shape[0])]
    ax.set_xticks(n_labels)
    ax.set_yticks(n_labels)
    ax.set_xticklabels(np.unique(y_test))
    ax.set_yticklabels(np.unique(y_test))

    ax.xaxis.set_ticks_position('bottom')
    plt.xlabel('Predict label')
    plt.ylabel('True Label')
    plt.show()





def validate(test_dl, model, loss_fn):
    model.eval()
    epoch_losses_test = []
    epoch_accuracy_test = []

    with torch.no_grad():
        for batch_inputs, batch_labels in test_dl:
            pred = model(batch_inputs) # [:, 0, :]
            loss = loss_fn(pred, batch_labels.long())
            epoch_losses_test.append(loss.item())

            pred = torch.nn.functional.softmax(pred, dim=1)
            accuracy = (pred.max(1, keepdim=True)[1][:, 0] == batch_labels).float()
            accuracy = accuracy.mean().item()
            epoch_accuracy_test.append(accuracy)
        return  np.mean(epoch_losses_test), np.mean(epoch_accuracy_test)




def plot_learning_curve(loss_hist_train, accuracy_hist_train, loss_hist_test, accuracy_hist_test):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(12, 10))
    ax0.plot(loss_hist_train, linestyle='solid', label='Training')
    ax0.plot(loss_hist_test, linestyle='solid', label='Test')
    ax0.set_title("Loss history")
    ax0.legend()

    ax1.plot(accuracy_hist_train, linestyle='solid', label='Training')
    ax1.plot(accuracy_hist_test, linestyle='solid', label='Test')
    ax1.set_title("Accuracy history")
    ax1.legend()

    plt.show()




class BananosEspectroDataset(Dataset):
    def __init__(self, inputs, labels):
        """
        inputs es un array de numpy de tamaño (n_samples, n_feature)
        labels es un array de numpy de tamaño (n_samples,)
        """

        self.inputs = torch.tensor(inputs.tolist(), dtype=torch.float32) # convertir a inputs en un tensor de tamaño torch.Size([n_samples, n_feature])
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long) # convertir a labels en un tensor de tamaño torch.Size([n_samples])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.inputs[idx].unsqueeze(0) # x es un tensor de tamaño torch.Size([1, n_feature])
        y = self.labels[idx] # y es un tensor de tamaño torch.Size([]) ejem: tensor(0.0655)
        return x, y


class EspectroModel(nn.Module):
    def __init__(self):
         super(EspectroModel, self).__init__()

         self.fc = nn.Sequential(
              nn.Linear(in_features=73, out_features=40),
              nn.ReLU(),
              nn.Linear(in_features=40, out_features=20),
              nn.ReLU(),
              nn.Linear(in_features=20, out_features=10),
              nn.ReLU(),
              nn.Linear(in_features=10, out_features=6),
              nn.ReLU(),
              nn.Linear(in_features=6, out_features=3),
              nn.Tanh()
         )
    
    def forward(self, x):
         x = self.fc(x)
         return x
    
class PruebaModel(nn.Module):
    def __init__(self):
        super(PruebaModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=12, stride=1, padding=2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=10, kernel_size=12, stride=2, padding=2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=12, kernel_size=12, stride=1, padding=2),
            nn.Tanh()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=14, kernel_size=12, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=16)
        )

        self.flatten = nn.Flatten()  # torch.Size([batch_size, 14])

        self.fc = nn.Sequential(
              nn.Linear(in_features=14, out_features=8),
              nn.ReLU(),
              nn.Linear(in_features=8, out_features=6),
              nn.Tanh(),
              nn.Linear(in_features=6, out_features=10),
              nn.Softplus(),
              nn.Linear(in_features=10, out_features=6),
              nn.ReLU(),
              nn.Linear(in_features=6, out_features=3),
              nn.Tanh()
         )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



class PruebaModel2(nn.Module):
    def __init__(self):
        super(PruebaModel2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=12, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=10, kernel_size=12, stride=1, padding=2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=12, kernel_size=12, stride=1, padding=2),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=14, kernel_size=12, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=14, out_channels=16, kernel_size=12, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=18, kernel_size=8, stride=1, padding=2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=15)
        )


        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
              nn.Linear(in_features=18, out_features=10),
              nn.ReLU(),
              nn.Linear(in_features=10, out_features=6),
              nn.Tanh(),
              nn.Linear(in_features=6, out_features=8),
              nn.Softplus(),
              nn.Linear(in_features=8, out_features=6),
              nn.ReLU(),
              nn.Linear(in_features=6, out_features=3),
              nn.Tanh()
         )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



def featureImportancia(importances, indices):
    x = [float(i) for i in indices]
    # dark_background    seaborn-whitegrid
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes()

    ax.plot(x, importances, color="#E73F9E",marker='o', markersize=3, linestyle='None',label=r"Pesos/%Frecuencias")

    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend()
    plt.show()