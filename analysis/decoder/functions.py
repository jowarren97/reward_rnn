import sys
sys.path.append('../../model/')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from config import Conf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

def get_anomalous_batches(choices, ignore_first_trial=True):
    start = 0 if not ignore_first_trial else Conf.trial_len
    
    action = choices[:,:,Conf.x_dim + Conf.r_dim:][:, Conf.ab_choice_step::Conf.trial_len]
    # Turn action array into 1's if element in row equals max in row
    action = np.argmax(action, axis=-1)

    anomalous_idxs = []
    for idx, row in enumerate(action):
        unique, counts = np.unique(row[start:], return_counts=True)
        if len(unique) > 2:
            anomalous_idxs.append(idx)

    non_anomalous_idxs = [i for i in range(len(action)) if i not in anomalous_idxs]
    print('# anomalous:', len(anomalous_idxs))
    print('# non-anomalous:', len(non_anomalous_idxs))

    return anomalous_idxs, non_anomalous_idxs


def swap_unique_elements(arr, layouts):
    if layouts.shape[1] != 2:
        raise ValueError('Layouts shape must be (n_batches, 2)')
    if layouts.shape[0] != arr.shape[0]:
        raise ValueError('Layouts shape[0] must match arr shape[0]')
    
    arr = arr.copy()
    for row, layout in zip(arr, layouts):
        a, b = layout[0], layout[1]
        a_idx = np.where(row == a)[0]
        b_idx = np.where(row == b)[0]

        row[a_idx], row[b_idx] = b, a

    return arr

def mask_conf_mats(conf_mats, bins, n_p_bins, mask='all'):
    if mask not in ['all', 'step']:
        raise ValueError('mask must be "all" or "step"')
    
    if mask == 'all': idx = np.array(list(set(np.arange(n_p_bins)) - set(np.unique(bins))))
    conf_mats_masked = []
    for conf_mat, bin in zip(conf_mats, bins):
        conf_mat_masked = conf_mat.copy()
        if mask == 'step': idx = np.array(list(set(np.arange(n_p_bins)) - set(np.unique(bin))))
        if len(idx):
            conf_mat_masked[idx,:] = np.nan
            conf_mat_masked[:,idx] = np.nan

        conf_mats_masked.append(conf_mat_masked)
    return conf_mats_masked

def plot_conf_matrices(conf_matrices, bins, n_bins, bin_edges=None, mask_type='all'):
    n_row, n_col = len(conf_matrices) // Conf.trial_len + 1, Conf.trial_len
    fig = plt.figure(figsize=(1.5 * n_col, 1.5 * n_row))
    gs = gridspec.GridSpec(n_row, n_col, figure=fig, wspace=0.1, hspace=0.1)

    conf_matrices_masked = mask_conf_mats(conf_matrices, bins, n_bins, mask=mask_type)
    for i, (m) in enumerate(conf_matrices_masked):
        cmap = plt.cm.get_cmap('viridis')
        cmap.set_bad('gray')  # Set the color for NaN values

        ax = fig.add_subplot(gs[i])
        ax.imshow(m, vmin=0, vmax=1, cmap=cmap, interpolation='none')

        if i % Conf.trial_len == 0:
            ax.set_ylabel('True')
        if i // Conf.trial_len == 0:
            ax.set_xlabel('Pred')
            ax.xaxis.set_label_position('top')
            ax.set_title(f'step {i}')

        ax.set_yticks([])
        ax.set_xticks([])

    bin_counts = np.bincount(bins.flatten(), minlength=n_bins)

    if bin_edges is not None:
        for i in range(Conf.trial_len):
            ax = fig.add_subplot(gs[-1, i])
            ax.bar(bin_edges - 0.5 / n_bins, bin_counts, width=np.diff(bin_edges, prepend=0), align='edge', capsize=2)
            ax.set_xlim([0, 1])
            ax.set_xlabel('p(A)')
            ax.set_xticks([i / 10 for i in range(10)], [])
            ax.yaxis.set_visible(False)
            if i == 0:
                ax.set_ylabel('count')

    plt.tight_layout()


def train_decoders(hiddens, targets, layouts, exclude_batch_idxs=None, n_class=None, T=None, **kwargs):
    if exclude_batch_idxs is not None:
        hiddens_masked = np.delete(hiddens.copy(), exclude_batch_idxs, axis=0)
        targets_masked = np.delete(targets.copy(), exclude_batch_idxs, axis=0)
        layouts_masked = np.delete(layouts.copy(), exclude_batch_idxs, axis=0)
    else:
        hiddens_masked = hiddens.copy()
        targets_masked = targets.copy()
        layouts_masked = layouts.copy()
    
    accuracies, conf_matrices, chance = [], [], []
    y_tests, y_preds = [], []

    if hiddens.ndim != 3:
        raise ValueError('hiddens must be a 3D array with shape (n_batches, n_steps, n_neurons)')
    if targets.ndim not in [1,2]:
        raise ValueError('targets must be a 1D or 2D array with shape (n_batches, ) or (n_batches, n_steps)')
    if targets.ndim == 2 and targets.shape[1] != hiddens.shape[1]:
        raise ValueError('targets shape[1] must match hiddens shape[1]')
    
    if T is None or T > hiddens_masked.shape[1]: T = hiddens_masked.shape[1]
    conf_labels = np.unique(targets_masked) if n_class is None else np.arange(n_class)

    if set(np.unique(targets_masked)).issubset([True, False]):
        multiclass = False
    else:
        multiclass = True
    print(f'Using multiclass: {multiclass}')


    for step in tqdm(range(T)):
        # y = A_good[:, step].ravel()
        if targets_masked.ndim == 1:
            y = targets_masked
        else:
            y = targets_masked[:,step].ravel()
        X = hiddens_masked[:,step]

        X_train, X_test, y_train, y_test, layouts_train, layouts_test = train_test_split(X, y, layouts_masked, test_size=0.2, random_state=40)

        if np.unique(y_test).size < 2:
            accuracies.append(np.nan)
            chance.append(np.nan)
            continue
        
        # # Optionally, standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define and train the logistic regression model
        model = LogisticRegression(solver='lbfgs', class_weight='balanced', **kwargs)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_tests.append(y_test)
        y_preds.append(y_pred)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=conf_labels, normalize='true')

        accuracies.append(accuracy)
        conf_matrices.append(conf_matrix)
        chance.append(1 / len(np.unique(y_test)))

        # print(f"Step {step} accuracy: {accuracy}")
        # print(f"Confusion Matrix:\n{conf_matrix}")
        # print(f"Classification Report:\n{report}")
    return accuracies, conf_matrices, chance, (layouts_test, np.array(y_tests), np.array(y_preds))
    

def train_decoders_looped(hiddens, targets, layouts, train_steps, T=None, scale=False, shuffle=False, n_class=None, fold_trials=True, exclude_batch_idxs=None, **kwargs):
    np.random.seed(None if 'random_state' not in kwargs else kwargs['random_state'])
    if fold_trials:
        if (len(train_steps) > Conf.trial_len) and (len(train_steps) % Conf.trial_len != 0):
            raise ValueError('Length of train_steps must be a multiple of Conf.trial_len')
        if list(train_steps) != list(range(min(train_steps), max(train_steps)+1)):
            raise ValueError('train_steps must be consecutive')
    
    if T is None or T > hiddens.shape[1]: T = hiddens.shape[1]

    if exclude_batch_idxs is not None:
        include_batch_idxs = np.array(list(set(range(hiddens.shape[0])) - set(exclude_batch_idxs)))
        print(len(include_batch_idxs), len(exclude_batch_idxs))
        hiddens_masked = hiddens.copy()[include_batch_idxs]
        targets_masked = targets.copy()[include_batch_idxs]
        layouts_masked = layouts.copy()[include_batch_idxs]
    else:
        hiddens_masked = hiddens.copy()
        targets_masked = targets.copy()
        layouts_masked = layouts.copy()

    conf_labels = np.unique(targets_masked) if n_class is None else np.arange(n_class)

    X_train, X_test, y_train, y_test, layout_train, layout_test = train_test_split(hiddens_masked, targets_masked, layouts_masked, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, layout_train.shape, layout_test.shape)

    X_train_sub = X_train[:, train_steps, :]
    y_train_sub = y_train[:, train_steps]

    print(X_train_sub.shape, y_train_sub.shape)

    if fold_trials:
        T_train = np.minimum(len(train_steps), Conf.trial_len)
        X_train_sub = X_train_sub.reshape(-1, T_train, X_train_sub.shape[-1])
        y_train_sub = y_train_sub.reshape(-1, T_train)
    else:
        T_train = len(train_steps)

    print(X_train_sub.shape, y_train_sub.shape)

    models, scalers = [], []
    for step in tqdm(range(T_train)):
        X = X_train_sub[:, step, :]
        y = y_train_sub[:, step]

        perm = np.random.permutation(X.shape[0])
        if shuffle: X, y = X[perm], y[perm]
        
        # # # Optionally, standardize features
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            scalers.append(scaler)
        else:
            scalers.append(None)

        # Define and train the logistic regression model
        model = LogisticRegression(solver='lbfgs', class_weight='balanced', **kwargs)
        model.fit(X, y)

        models.append(model)

    accuracies_all, conf_matrices_all = [], []
    for model, scaler in zip(models, scalers):
        accuracies, conf_matrices, y_tests, y_preds = [], [], [], []
        for step in tqdm(range(T)):
            X = X_test[:, step, :]
            y = y_test[:, step]

            # Optionally, standardize features
            if scale: X = scaler.transform(X)
            
            y_pred = model.predict(X)
            y_tests.append(y)
            y_preds.append(y_pred)

            # Evaluate the model
            accuracy = accuracy_score(y, y_pred)
            accuracies.append(accuracy)

            conf_matrix = confusion_matrix(y, y_pred, labels=conf_labels)
            conf_matrices.append(conf_matrix)

        accuracies_all.append(accuracies)
        conf_matrices_all.append(conf_matrices)

    return accuracies_all, conf_matrices_all
            # print(f"Step {step} accuracy: {accuracy}")
            # print(f"Confusion Matrix:\n{conf_matrix}")
            # print(f"Classification Report:\n{report}")


def train_decoders_looped_port(port, hiddens, targets, layouts, train_steps, T=None, scale=False, n_class=None, fold_trials=True, exclude_batch_idxs=None, **kwargs):
    if fold_trials:
        if (len(train_steps) > Conf.trial_len) and (len(train_steps) % Conf.trial_len != 0):
            raise ValueError('Length of train_steps must be a multiple of Conf.trial_len')
        if list(train_steps) != list(range(min(train_steps), max(train_steps)+1)):
            raise ValueError('train_steps must be consecutive')
    
    if T is None or T > hiddens.shape[1]: T = hiddens.shape[1]

    if exclude_batch_idxs is not None:
        include_batch_idxs = np.array(list(set(range(hiddens.shape[0])) - set(exclude_batch_idxs)))
        print(len(include_batch_idxs), len(exclude_batch_idxs))
        hiddens_masked = hiddens.copy()[include_batch_idxs]
        targets_masked = targets.copy()[include_batch_idxs]
        layouts_masked = layouts.copy()[include_batch_idxs]
    else:
        hiddens_masked = hiddens.copy()
        targets_masked = targets.copy()
        layouts_masked = layouts.copy()

    conf_labels = np.unique(targets_masked) if n_class is None else np.arange(n_class)

    port_idxs = np.where(layouts_masked == port)[0]
    not_port_idxs = np.where(layouts_masked != port)[0]

    X_train, X_test = hiddens_masked[port_idxs], hiddens_masked[not_port_idxs]
    y_train, y_test = targets_masked[port_idxs], targets_masked[not_port_idxs]
    layout_train, layout_test = layouts_masked[port_idxs], layouts_masked[not_port_idxs]

    # X_train, X_test, y_train, y_test, layout_train, layout_test = train_test_split(hiddens_masked, targets_masked, layouts_masked, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, layout_train.shape, layout_test.shape)

    X_train_sub = X_train[:, train_steps, :]
    y_train_sub = y_train[:, train_steps]

    print(X_train_sub.shape, y_train_sub.shape)

    if fold_trials:
        T_train = np.minimum(len(train_steps), Conf.trial_len)
        X_train_sub = X_train_sub.reshape(-1, T_train, X_train_sub.shape[-1])
        y_train_sub = y_train_sub.reshape(-1, T_train)
    else:
        T_train = len(train_steps)

    print(X_train_sub.shape, y_train_sub.shape)

    models, scalers = [], []
    for step in tqdm(range(T_train)):
        X = X_train_sub[:, step, :]
        y = y_train_sub[:, step]
        
        # # # Optionally, standardize features
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            scalers.append(scaler)
        else:
            scalers.append(None)

        # Define and train the logistic regression model
        model = LogisticRegression(solver='lbfgs', class_weight='balanced', **kwargs)
        model.fit(X, y)

        models.append(model)

    accuracies_all, conf_matrices_all = [], []
    for model, scaler in zip(models, scalers):
        accuracies, conf_matrices, y_tests, y_preds = [], [], [], []
        for step in tqdm(range(T)):
            X = X_test[:, step, :]
            y = y_test[:, step]

            # Optionally, standardize features
            if scale: X = scaler.transform(X)
            
            y_pred = model.predict(X)
            y_tests.append(y)
            y_preds.append(y_pred)

            # Evaluate the model
            accuracy = accuracy_score(y, y_pred)
            accuracies.append(accuracy)

            conf_matrix = confusion_matrix(y, y_pred, labels=conf_labels)
            conf_matrices.append(conf_matrix)

        accuracies_all.append(accuracies)
        conf_matrices_all.append(conf_matrices)

    return accuracies_all, conf_matrices_all
            # print(f"Step {step} accuracy: {accuracy}")
            # print(f"Confusion Matrix:\n{conf_matrix}")
            # print(f"Classification Report:\n{report}")



def plot_decoding_accuracy(accuracies, chance=None, T=None, n_class=None, title=None):
    T = len(accuracies) if T is None else T

    if chance is not None:
        plt.plot(chance * np.ones(T), 'r--', label='Chance')
    elif n_class is not None:
        chance = 1 / n_class
        plt.hlines(chance, 0, T, 'r', 'dashed', label='Chance')
    plt.plot(accuracies, 'b', linewidth=2, label='Decoding accuracy')
    plt.xticks(range(0,T,Conf.trial_len), range(0,T,Conf.trial_len))
    plt.xlim([0, T])
    plt.ylim([0, 1.01])
    plt.grid(True, 'major')
    plt.ylabel('Decoding accuracy')
    plt.xlabel('Step')
    if title is not None: plt.title(title)
    plt.legend()

def plot_decoding_accuracy_looped(accs, train_steps, n_class=None, T=None, title=None):
    plt.figure()
    # Define the colormap
    cmap = plt.cm.get_cmap('viridis')

    T = len(accs[0]) if T is None else np.minimum(T, len(accs[0]))
    
    if n_class is not None:
        chance = 1 / n_class
        plt.hlines(chance, 0, T, 'r', 'dashed', label='Chance')

    # k=(np.array(train_steps[:Conf.trial_len])%Conf.trial_len)
    for i, (acc, train_step) in enumerate(zip(accs, train_steps)):
        # Calculate the color based on the index
        color = cmap(i / len(accs))
        plt.plot(acc, label=r'step $n + {train_step}$', color=color, marker='o', markersize=3)

    plt.xticks(np.arange(len(accs[0])))
    plt.xlim(left=0, right=T)
    plt.ylim([0, 1.01])
    plt.ylabel('Decoding accuracy')
    plt.xlabel('Step')
    if title is not None: plt.title(title)
    plt.legend()

def plot_decoding_accuracy_looped(accs, train_steps, n_class=None, T=None, title=None):
    plt.figure()
    # Define the colormap
    cmap = plt.cm.get_cmap('viridis')

    T = len(accs[0]) if T is None else np.minimum(T, len(accs[0]))
    
    if n_class is not None:
        chance = 1 / n_class
        plt.hlines(chance, 0, T, 'r', 'dashed', label='Chance')

    # Add alternating grey and white vertical bands
    for i in range(0, T, Conf.trial_len):
        if i // Conf.trial_len % 2 == 0:
            plt.axvspan(i - 0.5, i + Conf.trial_len - 0.5, facecolor='lightgrey', alpha=0.5)

    # k=(np.array(train_steps[:Conf.trial_len])%Conf.trial_len)
    for i, (acc, train_step) in enumerate(zip(accs, train_steps)):
        # Calculate the color based on the index
        color = cmap(i / len(accs))
        plt.plot(acc, label=f'step {train_step}', color=color, marker='o', markersize=3)

    plt.xticks(np.arange(len(accs[0])))
    plt.xlim(left=0, right=T)
    plt.ylim([0, 1.01])
    plt.ylabel('Decoding accuracy')
    plt.xlabel('Step')
    if title is not None: plt.title(title)
    plt.legend()
    

def normalise(arr, axis=1):
    row_sums = np.sum(arr, axis=axis, keepdims=True)
    row_sums[row_sums == 0] = 1  # Replace 0 with 1 to avoid division by zero
    return arr / row_sums

def plot_conf_matrices_by_step(conf_matrices):
    fig, axes = plt.subplots(Conf.trial_len, Conf.trial_len)

    for row, m in enumerate(conf_matrices):
        for col in range(Conf.trial_len):
            ax = axes[row, col]
            mats = m[col::Conf.trial_len]
            # Assuming you have a list of arrays called 'arr_list'
            mat = np.sum(mats, axis=0)
            mat = normalise(mat)
            ax.imshow(mat, vmin=0, vmax=1, cmap='viridis', interpolation='none')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    # Add major x label to the side of the figure
    fig.text(0.5, -0.04, 'Decoder tested on step', ha='center')
    # Add major y label to the side of the figure
    fig.text(0.04, 0.5, 'Decoder trained on step', va='center', rotation='vertical')
    fig.tight_layout()

"""
def train_decoders_looped(hiddens, targets, layouts, train_steps=None, exclude_batch_idxs=None, n_class=None, T=None, **kwargs):
    if exclude_batch_idxs is not None:
        include_batch_idxs = np.array(list(set(range(hiddens.shape[0])) - set(exclude_batch_idxs))
        hiddens_masked = hiddens.copy()[include_batch_idxs]
        targets_masked = targets.copy()[include_batch_idxs]
        layouts_masked = layouts.copy()[include_batch_idxs]
    else:
        hiddens_masked = hiddens.copy()
        targets_masked = targets.copy()
        layouts_masked = layouts.copy()
    
    accuracies, conf_matrices, chance = [], [], []
    y_tests, y_preds = [], []

    if hiddens.ndim != 3:
        raise ValueError('hiddens must be a 3D array with shape (n_batches, n_steps, n_neurons)')
    if targets.ndim not in [1,2]:
        raise ValueError('targets must be a 1D or 2D array with shape (n_batches, ) or (n_batches, n_steps)')
    
    if T is None or T > hiddens_masked.shape[1]: T = hiddens_masked.shape[1]
    conf_labels = np.unique(targets_masked) if n_class is None else np.arange(n_class)
    print(conf_labels)
    print(T)

    if set(np.unique(targets_masked)).issubset([True, False]):
        multiclass = False
    else:
        multiclass = True
    print(f'Using multiclass: {multiclass}')


    for step in tqdm(range(T)):
        # y = A_good[:, step].ravel()
        if targets_masked.ndim == 1:
            y = targets_masked
        else:
            y = targets_masked[:,step].ravel()
        X = hiddens_masked[:,step]

        print(X.shape, y.shape)
        X_train, X_test, y_train, y_test, layouts_train, layouts_test = train_test_split(X, y, layouts_masked, test_size=0.2, random_state=40)

        if np.unique(y_test).size < 2:
            accuracies.append(np.nan)
            chance.append(np.nan)
            continue
        
        # # Optionally, standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define and train the logistic regression model
        model = LogisticRegression(solver='lbfgs', class_weight='balanced', **kwargs)
        model.fit(X_train_scaled, y_train)

        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_tests.append(y_test)
        y_preds.append(y_pred)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred, labels=conf_labels, normalize='true')

        accuracies.append(accuracy)
        conf_matrices.append(conf_matrix)
        chance.append(1 / len(np.unique(y_test)))




import torch

b = 154
num_trials = 80
t = num_trials * Conf.trial_len

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(10, num_trials))

def display_mat(ax, data, title='', highlight=False):
    data = data.clone().numpy() if type(data) == torch.Tensor else data
    ax.imshow(data, aspect='auto', interpolation='none')
    ax.axvline(x=Conf.x_dim-0.5, c='w', linewidth=1)
    ax.axvline(x=Conf.r_dim + Conf.x_dim-0.5, c='w', linewidth=1)
    for i in range(1, t//Conf.trial_len + 1):
        ax.axhline(y=i * Conf.trial_len-0.5, c='w', linewidth=1)
    ax.set_title(title, fontsize=10)

    if highlight:
        for i, choice in enumerate(np.argmax(data, axis=-1)):
            rect = patches.Rectangle((choice-0.5, i-0.5), 1, 1, linewidth=1, edgecolor='white', facecolor='none')
            axes[3].add_patch(rect)

    # ax.yaxis.set_visible(False)
    # ax.xaxis.set_visible(False)
    # q: how do I turn on the grid on the plot?
    # a: ax.grid(True)
    # q: its not showing up
    # a: try plt.grid(True)
    # q: its still not showing up
    # Set grid
    ax.set_xticks([i+0.5 for i in range(data.shape[1])])  # Adjust x-ticks to align with grid
    ax.set_yticks([i+0.5 for i in range(data.shape[0])])  # Adjust y-ticks to align with grid
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    ax.grid(True, which='both', color='white', alpha=0.3, linestyle='-', linewidth=0.5)

a_idx = Conf.x_dim + Conf.r_dim
display_mat(axes[0], inputs[b, :t, a_idx:], title='inputs')
display_mat(axes[1], ground_truth[b, :t, a_idx:], title='groundtruths')
display_mat(axes[2], targets[b, :t, a_idx:], title='targets')
display_mat(axes[3], choices[b, :t, a_idx:], title='choices', highlight=True)
# display_mat(axes[3], softmax(choices[b, :t, a_idx:], axis=-1), title='choices')
print(np.argmax(choices[b, :t, a_idx:], axis=-1))
"""