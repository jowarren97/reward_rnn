from functions.functions_cells_policy import task_ind
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import logistic
from scipy.signal import butter, sosfilt, filtfilt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import os
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def glm_input_data(data, n, use_choices=False):
    """ CHECKED AND FINE """
    dm = data['DM'][0]
    X_sessions, y_sessions, task_id_sessions = [], [], []
    # for  s, sess in enumerate(dm):     
    #     DM = dm[s]; choices = DM[:,1]; reward = DM[:,2]; task =  DM[:,5];  a_pokes = DM[:,6]; b_pokes = DM[:,7]
    #     taskid = task_ind(task, a_pokes, b_pokes)[n:]

    #     previous_rewards = scipy.linalg.toeplitz(reward, np.zeros((1,n)))[n-1:-1] # find rewards on n-back trials
    #     previous_choices = scipy.linalg.toeplitz(choices, np.zeros((1,n)))[n-1:-1] # find choices on n-back trials
    #     interactions = scipy.linalg.toeplitz((((choices-0.5)*(reward-0.5))*2),np.zeros((1,n)))[n-1:-1] # interactions rewards x choice
    #     choices_current = (choices[n:]) # current choices need to start at nth trial
    #     ones = np.ones(len(interactions)).reshape(len(interactions),1) # add constant
    #     X = (np.hstack([interactions, ones])) if not use_choices else (np.hstack([previous_choices, interactions, ones])) # create design matrix

    #     X_sessions.append(X)
    #     y_sessions.append(choices_current)
    #     task_id_sessions.append(taskid)

    for  s, sess in enumerate(dm):
        DM = dm[s]; choices = DM[:,1]; reward = DM[:,2]; task =  DM[:,5];  a_pokes = DM[:,6]; b_pokes = DM[:,7]
        task_ids = task_ind(task, a_pokes, b_pokes)
        X_tasks, y_tasks, task_id_tasks = [], [], []
        for i in [1,2,3]:
            task_mask = task_ids == i
            reward_masked = reward[task_mask]; choices_masked = choices[task_mask]; task_ids_masked = task_ids[task_mask]
        
            task_ids_masked = task_ids_masked[n:]
            previous_rewards = scipy.linalg.toeplitz(reward_masked, np.zeros((1,n)))[n-1:-1] # find rewards on n-back trials
            previous_choices = scipy.linalg.toeplitz(choices_masked, np.zeros((1,n)))[n-1:-1] # find choices on n-back trials
            interactions = scipy.linalg.toeplitz((((choices_masked-0.5)*(reward_masked-0.5))*2),np.zeros((1,n)))[n-1:-1] # interactions rewards x choice
            choices_current = (choices_masked[n:]) # current choices need to start at nth trial
            ones = np.ones(len(interactions)).reshape(len(interactions),1) # add constant
            X = (np.hstack([interactions, ones])) if not use_choices else (np.hstack([previous_choices, interactions, ones])) # create design matrix

            X_tasks.append(X); y_tasks.append(choices_current); task_id_tasks.append(task_ids_masked)
            
        X_tasks = np.concatenate(X_tasks); y_tasks = np.concatenate(y_tasks); task_id_tasks = np.concatenate(task_id_tasks)
        X_sessions.append(X_tasks); y_sessions.append(y_tasks); task_id_sessions.append(task_id_tasks)
    
    return X_sessions, y_sessions, task_id_sessions

def glm_train(X, Y, random_seed):
    model = LogisticRegression(fit_intercept = False) 

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)
    model.fit(X_train, Y_train) # fit logistic regression predicting current choice based on history of behaviour

    # Evaluate the model
    y_pred = model.predict(X_train)
    accuracy_train = 100*accuracy_score(Y_train, y_pred)
    y_pred = model.predict(X_test)
    accuracy_test = 100*accuracy_score(Y_test, y_pred)
    print(f'Train : {accuracy_train:.1f}\nTest : {accuracy_test:.1f}')

    model = LogisticRegression(fit_intercept = False) 
    model.fit(X, Y) # fit logistic regression predicting current choice based on history of behaviour

    return model

def glm_value(model, X, n):
    value_coef = model.coef_[0][-n-1:-1]
    value = np.matmul(X[:, -n-1:-1], value_coef)
    return value

def glm_values(data, n=11, use_choices=True, random_seed=0):
   # Train model on concatenated data 
    X_sessions, y_sessions, taskid_sessions = glm_input_data(data, n, use_choices=use_choices)
    X, Y = np.vstack(X_sessions), np.concatenate(y_sessions)
    model = glm_train(X, Y, random_seed=random_seed)

    values = []
    values_task1, values_task2, values_task3 = [], [], []
    # Run the model on list data
    for x, taskid in zip(X_sessions, taskid_sessions):
        v = glm_value(model, x, n)
        values.append(v)

        v1 = glm_value(model, x[taskid==1], n); values_task1.append(v1)
        v2 = glm_value(model, x[taskid==2], n); values_task2.append(v2)
        v3 = glm_value(model, x[taskid==3], n); values_task3.append(v3)

    return values, [values_task1, values_task2, values_task3]

def trim_data(PFC, n):
    firing_rates = PFC['Data'][0]; dm = PFC['DM'][0]
    firing_rates_sessions = []
    for session, firing_rate in enumerate(firing_rates):
        DM = dm[session]; task =  DM[:,5];  a_pokes = DM[:,6]; b_pokes = DM[:,7]
        task_ids = task_ind(task, a_pokes, b_pokes)
        firing_rate_session = []
        for task_id in [1,2,3]: 
            task_mask = task_ids == task_id
            firing_rate_masked = firing_rate[task_mask]
            firing_rate_masked = firing_rate_masked[n:]
            firing_rate_session.append(firing_rate_masked)
        firing_rate_session = np.concatenate(firing_rate_session, axis=0)
        firing_rates_sessions.append(firing_rate_session)
    return firing_rates_sessions

def value_transition_matrix(values, n_p_bins=50, use_tm=True):
    counts = []
    for v in values:
        # if use_sig: 
        #     belief = logistic.cdf(v)
        #     if fold: belief = 0.5 + np.abs(belief - 0.5)
        #     lims = [0, 1] if not fold else [0.5, 1]
        #     bin_edges = np.linspace(lims[0], lims[1], n_p_bins+1)[1:-1]  # Creates 5 edges for 4 bins
        #     bins = np.digitize(belief, bin_edges)
        lims = [-np.max(np.abs(v)), np.max(np.abs(v))]
        bin_edges = np.linspace(lims[0], lims[1], n_p_bins+1)[1:-1]
        bins = np.digitize(v, bin_edges)

        # Number of unique states
        bins = bins.flatten()
        num_states = n_p_bins

        # Collect transitions
        rows = bins[:-1]; cols = bins[1:]
        # Count transitions using a sparse matrix
        count = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_states, num_states)).toarray()
        counts.append(count)
    # Normalize to get probabilities
    counts = np.sum(counts, axis=0)
    transition_matrix = counts / counts.sum(axis=1, keepdims=True)
    transition_matrix[np.isnan(transition_matrix)] = 0

    M = transition_matrix if use_tm else counts
    plt.figure(figsize=(3,3), dpi=150)
    plt.imshow(M, vmin=0, vmax=1 if use_tm else np.max(M), cmap='viridis')
    plt.xticks(np.arange(0, n_p_bins, 10), [f'{b:.2f}' for b in bin_edges[::10]], fontsize=8)
    plt.yticks(np.arange(0, n_p_bins, 10), [f'{b:.2f}' for b in bin_edges[::10]], fontsize=8)
    # plt.ylabel(r'$p(good\ port)_{t}$')
    # plt.xlabel(r'$p(good\ port)_{t+1}$')
    plt.ylabel(r'$V_{t}$')
    plt.xlabel(r'$V_{t+1}$')
    # plt.xlim(lims); plt.ylim(lims)
    plt.colorbar()

    return M

    # eigenvalues, eigenvectors = np.linalg.eig(M if use_tm else M / np.max(M))

    # print(eigenvectors.shape)
    # n = 49
    # fig, axes = plt.subplots(7,7, figsize=(10,10), dpi=300)
    # fig.suptitle('Eigenvectors of transition probability matrix' if use_tm else 'Eigenvectors of transition counts matrix')
    # for i, ax in enumerate(axes.flatten()):
    #     v = eigenvectors[:, i]
    #     ax.bar(np.arange(len(v)), v, width=1)
    #     ax.set_ylim([-1.2*np.max(np.abs(v)), 1.2*np.max(np.abs(v))])
    #     ax.xaxis.set_visible(False)
    #     ax.yaxis.set_visible(False)
    #     ax.set_title(f'{eigenvalues[i]:.2f}')
    #     ax.set_xlim([-0.5, n_p_bins-0.5])

    # plt.tight_layout()

def sort_firing_by_bins(PFC, values, task_id=None, n=None, n_p_bins=100):
    firing_by_bins_sessions = []
    firing_rates = PFC['Data'][0]; dm = PFC['DM'][0]
   
    for session, (value, firing_rate) in enumerate(zip(values, firing_rates)):
        DM = dm[session]; task =  DM[:,5];  a_pokes = DM[:,6]; b_pokes = DM[:,7]
        task_ids = task_ind(task, a_pokes, b_pokes)
        if task_id is not None:
            task_mask = task_ids == task_id
            firing_rate = firing_rate[task_mask]
        if n is None:
            n = firing_rate.shape[0] - value.shape[0]
            print('n: ', n)
        firing_rate = firing_rate[n:]

        lims = [-np.max(np.abs(value)), np.max(np.abs(value))]
        bin_edges = np.linspace(lims[0], lims[1], n_p_bins+1)[1:-1]  # Creates 5 edges for 4 bins
        bins = np.digitize(value, bin_edges)
        bins = bins.flatten()

        sample = np.zeros(n_p_bins)
        firing_by_bins = []
        print(value.shape, bins.shape, firing_rate.shape)
        for p in range(n_p_bins):
            p_mask = bins == p
            p_firing_rate = firing_rate[p_mask]
            firing_by_bins.append(p_firing_rate)
            sample[p] = np.count_nonzero(p_mask)
        
        print([x.shape for x in firing_by_bins])
        firing_by_bins_sessions.append(firing_by_bins)

    return firing_by_bins_sessions


def get_tuning(firing, t):
    firing_by_bins_timewindow = [z[:, :, t] for z in firing]
    tuning = np.array([np.nanmean(x, axis=(0,2)) for x in firing_by_bins_timewindow])  # average across sample and time
    std = np.array([np.nanstd(x, axis=(0,2)) for x in firing_by_bins_timewindow])

    tuning = tuning.transpose(1,0)
    std = std.transpose(1,0)
    
    samples = np.array([x.shape[0] * x.shape[2] for x in firing_by_bins_timewindow])

    scatter = [f.transpose(1,0,2).reshape(f.shape[1], -1) for f in firing_by_bins_timewindow]

    return tuning, std, samples, scatter

def smooth_upsample(data, factor, sigma):
    n_p_bins = len(data)
    x_old = np.linspace(0, n_p_bins-1, n_p_bins)  # Original indices
    x_new = np.linspace(0, n_p_bins-1, n_p_bins*factor)  # New indices for higher resolution
    valid = ~np.isnan(data)
    interpolator = interp1d(x_old[valid], data[valid], kind='zero', fill_value="extrapolate")
    upsampled_data = interpolator(x_new)
    smoothed_upsampled_data = gaussian_filter(upsampled_data, sigma=sigma)
    return smoothed_upsampled_data

def smooth_upsample_tunings(data, factor=5, sigma=5):
    return np.array([smooth_upsample(d, factor, sigma) for d in data])

def high_pass_filter(data, cutoff_frequency, fs, order=10):
    """
    Apply a high-pass filter to an array along the second dimension.

    Parameters:
    - data: numpy array of shape (n_neurons, n_bins)
    - cutoff_frequency: cutoff frequency of the high-pass filter in Hz
    - fs: sampling frequency in Hz
    - order: order of the filter

    Returns:
    - filtered_data: numpy array of shape (n_neurons, n_bins) with the filter applied
    """
    # Create a Butterworth high-pass filter
    b, a = butter(order, cutoff_frequency, fs=fs, btype='high', output='ba')
    # Apply the filter to each row (neuron)
    # filtered_data = np.zeros_like(data)
    # if data.ndim == 1:
    #     filtered_data = filtfilt(b, a, data)
    # else:
    #     for i, x in enumerate(data):
    #         filtered_data[i] = filtfilt(b, a, x)

    filtered_data = filtfilt(b, a, data, axis=-1)
    # filtered_data = np.apply_along_axis(lambda x: filtfilt(b, a, x), axis=1, arr=data)
    return filtered_data

def detrend_linear(data, keep_dc):
    """
    Remove linear trend from each row (neuron) in the data array.

    Parameters:
    - data: numpy array of shape (n_neurons, n_bins)

    Returns:
    - detrended_data: numpy array of shape (n_neurons, n_bins)
    """
    n_neurons, n_bins = data.shape
    detrended_data = np.empty_like(data)
    dc = np.mean(data, axis=1)
    # Apply linear detrending
    for i in range(n_neurons):
        # Create an array of indices (time points)
        t = np.arange(n_bins)
        # Fit a linear trend (y = mx + b) and subtract it
        p = np.polyfit(t, data[i], 1)  # p[0] is slope, p[1] is intercept
        trend = p[0] * t + p[1]
        detrended_data[i] = data[i] - trend
    
    if keep_dc: detrended_data += dc[:, np.newaxis]
    return detrended_data


def plot_value_grid(PFC, values, phase, task_id=None, n_fig=1, n_p_bins=100, plot_scatter=False, use_smoothed=True, sigma=8, use_color=True, save_dir=None):
    time_ind = [0, 24, 35, 42, 63]
    t = np.arange(time_ind[phase]+1, time_ind[phase+1])
    
    save_dir = os.path.join('./analysis/neurons/grids/', f'phase_{str(phase)}', 'smoothed' if use_smoothed else 'raw', 'color' if use_color else 'bw', f'sigma_{sigma}', f'{n_p_bins}bins', 'ugh3')
    os.makedirs(save_dir, exist_ok=True)
    
    # cmap = plt.cm.get_cmap('viridis')
    cmap = plt.cm.get_cmap('jet')

    firing_by_bins_sessions = sort_firing_by_bins(PFC, values, task_id, n_p_bins=n_p_bins)

    for session, firing_by_bins in enumerate(firing_by_bins_sessions[:n_fig]):
        tuning_mean, tuning_std, sample, scatter = get_tuning(firing_by_bins, t)
        n_neurons = tuning_mean.shape[0]
        fig, axes = plt.subplots(1+n_neurons, 1, figsize=(6, 1.3*(1+n_neurons)), dpi=160)

        if len(t) > 1:
            fig.suptitle(f'PFC session {session} - timepoints {t[0]}-{t[-1]}', fontsize=16, y=1.005)
        else:
            fig.suptitle(f'PFC session {session} - timepoint {t}', fontsize=16)

        ax = axes[0]
        ax.bar(np.arange(len(sample)), sample, color='k', width=1, edgecolor='black')
        ax.title.set_text(f'# Samples')
        ax.xaxis.set_major_locator(MultipleLocator(len(sample)/4))
        ax.xaxis.set_minor_locator(MultipleLocator(len(sample)/16))
        ax.set_xticklabels([])

        tuning_smoothed = smooth_upsample_tunings(tuning_mean, factor=5, sigma=sigma)
        # tuning_smoothed = detrend_linear(tuning_smoothed, keep_dc=True)
        # tuning_smoothed = high_pass_filter(tuning_smoothed, cutoff_frequency=0.01, fs=tuning_smoothed.shape[1], order=10)
        # q: how can I high-pass filter the tuning curves?
        # a: use a gaussian filter with a large sigma
        
        y = tuning_mean if not use_smoothed else tuning_smoothed
        
        for j in range(n_neurons):
            vmin = np.nanmin(tuning_mean[j,:]); vmax = np.nanmax(tuning_mean[j,:])
            # vmin = np.nanmin(y[j,:]); vmax = np.nanmax(y[j,:])
            ax = axes[j+1]
            plot_inset_tuning(ax, y[j], vmin, vmax)

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f'session{session}.png'), dpi=300)
        if n_fig > 1: plt.close()

    return tuning_smoothed

def row_correlations(array1, array2):
    """
    Calculate the correlation coefficients between corresponding rows of two arrays.
    
    Parameters:
    - array1, array2: numpy arrays of the same shape (n_rows, n_columns)
    
    Returns:
    - correlations: numpy array of correlation coefficients for corresponding rows
    """
    if array1.shape != array2.shape:
        raise ValueError("Both arrays must have the same shape.")
    
    n_rows = array1.shape[0]
    correlations = np.zeros(n_rows)
    
    for i in range(n_rows):
        # Extract the rows
        row1 = array1[i, :]
        row2 = array2[i, :]
        
        # Compute the correlation coefficient between the two rows
        # corrcoef returns a 2x2 matrix, we are interested in the off-diagonal elements [0,1] or [1,0]
        corr_matrix = np.corrcoef(row1, row2)
        correlations[i] = corr_matrix[0, 1]  # Take the correlation between the first and second row
    
    return correlations

def sample_from_histogram(correlations, num_samples=1):
    """
    Sample values from the distribution represented by the histogram of correlations.
    
    Parameters:
    - correlations: 1D numpy array of correlation values
    - num_samples: int, number of samples to generate
    
    Returns:
    - samples: numpy array of sampled correlation values
    """
    # Compute the histogram
    counts, bin_edges = np.histogram(correlations, bins=100, density=True)
    
    # Generate a probability mass function (PMF) from counts
    pmf = counts / np.sum(counts)
    
    # Choose bins based on PMF
    bins = np.random.choice(len(bin_edges)-1, size=num_samples, p=pmf)
    
    # Sample uniformly from each chosen bin
    sampled_values = bin_edges[bins] + np.random.rand(num_samples) * (bin_edges[bins+1] - bin_edges[bins])
    
    return sampled_values

def permute_value_grid(PFC, values_all, phase, n_p_bins=100, remove_linear=True, n_fig=1, sigma=8):
    time_ind = [0, 24, 35, 42, 63]
    t = np.arange(time_ind[phase]+1, time_ind[phase+1])
    
    save_dir = os.path.join('./analysis/neurons/grids/', f'phase_{str(phase)}', 'p_tests', f'phase_{phase}', f'sigma_{sigma}')
    os.makedirs(save_dir, exist_ok=True)

    firing_by_bins_by_task_sessions = []
    for task_id in [1,2,3]:
        firing_by_bins_sessions = sort_firing_by_bins(PFC, values_all[task_id-1], task_id, n_p_bins=n_p_bins)
        firing_by_bins_by_task_sessions.append(firing_by_bins_sessions)

    # values = [np.concatenate([x,y,z]) for x,y,z in zip(values_all[0], values_all[1], values_all[2])]
    firing_by_bins_sessions_all_tasks = []
    for f1_sess, f2_sess, f3_sess in zip(firing_by_bins_by_task_sessions[0], firing_by_bins_by_task_sessions[1], firing_by_bins_by_task_sessions[2]):
        f_sess = []
        for b, (f1_sess_bin, f2_sess_bin, f3_sess_bin) in enumerate(zip(f1_sess, f2_sess, f3_sess)):
            # print('1: ', f1_sess_bin.shape, f2_sess_bin.shape, f3_sess_bin.shape)
            f_sess_bin = np.concatenate([f1_sess_bin, f2_sess_bin, f3_sess_bin], axis=0)
            # print('2: ', f_sess_bin.shape)
            f_sess.append(f_sess_bin)
        firing_by_bins_sessions_all_tasks.append(f_sess)

    perms = [(1,2), (1,3), (2,3)]

    n_sessions = len(firing_by_bins_sessions)
    p_vals, p_vals_prod = [], []
        # for session, (firing_by_bins_1, firing_by_bins_2) in enumerate(zip(firing_by_bins_sessions_1[:n_fig], firing_by_bins_sessions_2[:n_fig])): 
    for session in range(np.minimum(n_fig, n_sessions)):  
        n_neurons = firing_by_bins_sessions[session][0].shape[1]
        print('n_neurons', n_neurons)
        fig, axes = plt.subplots(n_neurons, 5, figsize=(9, n_neurons), dpi=160)
        fig.suptitle(f'Session {session}', fontsize=16, y=1.005)
        
        tuning_mean = get_tuning(firing_by_bins_sessions_all_tasks[session], t)[0]
        tuning_smoothed = smooth_upsample_tunings(tuning_mean, factor=5, sigma=sigma)
        # tuning_smoothed = detrend_linear(tuning_smoothed, keep_dc=True)

        permuted_correlations = np.zeros((3, n_neurons, 5000)); true_correlations = np.zeros((3, n_neurons))
        p_vals_session = np.zeros((3, n_neurons))
        for i, perm in enumerate(perms):
            firing_by_bins_1 = firing_by_bins_by_task_sessions[perm[0]-1][session]
            firing_by_bins_2 = firing_by_bins_by_task_sessions[perm[1]-1][session]       
            tuning_mean_1, _, _, _ = get_tuning(firing_by_bins_1, t)            
            tuning_mean_2, _, _, _ = get_tuning(firing_by_bins_2, t)            
            tuning_mean_permuted_1 = tuning_mean_1.copy(); tuning_mean_permuted_2 = tuning_mean_2.copy()

            tuning_smoothed_1 = smooth_upsample_tunings(tuning_mean_1, factor=5, sigma=sigma)
            tuning_smoothed_2 = smooth_upsample_tunings(tuning_mean_2, factor=5, sigma=sigma)

            if remove_linear: tuning_smoothed_1 = detrend_linear(tuning_smoothed_1, keep_dc=False)
            if remove_linear: tuning_smoothed_2 = detrend_linear(tuning_smoothed_2, keep_dc=False)

            true_correlations[i] = row_correlations(tuning_smoothed_1, tuning_smoothed_2)

            for p in range(5000):
                # Shuffle each row in the copy
                for row in tuning_mean_permuted_1: np.random.shuffle(row)
                for row in tuning_mean_permuted_2: np.random.shuffle(row)

                tuning_permuted_smoothed_1 = smooth_upsample_tunings(tuning_mean_permuted_1, factor=5, sigma=sigma)
                tuning_permuted_smoothed_2 = smooth_upsample_tunings(tuning_mean_permuted_2, factor=5, sigma=sigma)

                if remove_linear: tuning_permuted_smoothed_1 = detrend_linear(tuning_permuted_smoothed_1, keep_dc=False)
                if remove_linear: tuning_permuted_smoothed_2 = detrend_linear(tuning_permuted_smoothed_2, keep_dc=False)

                permuted_correlations[i,:,p] = row_correlations(tuning_permuted_smoothed_1, tuning_permuted_smoothed_2)

            for j in range(n_neurons):
                ax = axes[j,i+1] if n_neurons > 1 else axes[i+1]
                ax.hist(permuted_correlations[i,j], bins=100, color='k', alpha=0.5)
                ax.axvline(true_correlations[i,j], color='r')
                # q: draw a vertical black line at 95% confidence interval
                # a: use np.percentile(permuted_correlations[j], 2.5) and np.percentile(permuted_correlations[j], 97.5)
                ax.axvline(np.percentile(permuted_correlations[i,j,:], 95), color='k', linestyle='--')
                if i == 1: ax.set_title(f'Neuron {j}', fontsize=8)
                ax.set_xlim([-1, 1])
                ax.set_xticklabels([])
                # for j in range(n_neurons):
                #     # vmin = np.nanmin(y[j,:]); vmax = np.nanmax(y[j,:])
                #     for i in range(n_bins):
                #         pass
                # q: how do I get the p value for each neuron (the true correlation)?
                p_vals_session[i,j] = np.sum(permuted_correlations[i,j] > true_correlations[i,j]) / permuted_correlations.shape[-1]
                # q: add text f'p = {p_vals_session[i,j]:.3f}' to the plot in the top right corner
                # a: use ax.text(x, y, text)
                # a: how do I position it top right
                ax.text(0.025, 0.95, f'p = {p_vals_session[i,j]:.3f}', 
                        transform=ax.transAxes, 
                        ha='left', va='top', 
                        fontsize=6,  # Optional, adjust as needed
                        color='blue'  # Optional, adjust as needed
                    )
        
        final_ps = np.zeros(n_neurons)
        for j in range(n_neurons):
            vmin = np.nanmin(tuning_mean[j,:]); vmax = np.nanmax(tuning_mean[j,:])
            ax = axes[j,0] if n_neurons > 1 else axes[0]
            plot_inset_tuning(ax, tuning_smoothed[j], vmin, vmax)

            corr_1_2 = sample_from_histogram(permuted_correlations[0,j,:], num_samples=10000)
            corr_1_3 = sample_from_histogram(permuted_correlations[1,j,:], num_samples=10000)
            corr_2_3 = sample_from_histogram(permuted_correlations[2,j,:], num_samples=10000)
            p_1_2 = np.sum(permuted_correlations[0,j][np.newaxis,:] > corr_1_2[:,np.newaxis], axis=1) / permuted_correlations.shape[-1]
            p_1_3 = np.sum(permuted_correlations[1,j][np.newaxis,:] > corr_1_3[:,np.newaxis], axis=1) / permuted_correlations.shape[-1]
            p_2_3 = np.sum(permuted_correlations[2,j][np.newaxis,:] > corr_2_3[:,np.newaxis], axis=1) / permuted_correlations.shape[-1]

            p_prod = p_1_2 * p_1_3 * p_2_3
            ax = axes[j,4] if n_neurons > 1 else axes[4]

            p_prod_true = p_vals_session[0,j] * p_vals_session[1,j] * p_vals_session[2,j]

            # q: make the x axis logarithmic and the binning logarithmic - or is this done automatically?
            ax.set_xscale('log')
            ax.hist(p_prod, bins=np.logspace(np.log10(0.01**3), np.log10(1), 100))
            # ax.hist(p_prod, bins=100, color='k', alpha=0.5)
            ax.axvline(np.percentile(p_prod, 5), color='k', linestyle='--')
            ax.axvline(p_prod_true, color='r')

            final_ps[j] = np.sum(p_prod < p_prod_true) / p_prod.shape[-1]

            ax.text(0.025, 0.95, 
                f'prod = {p_prod_true:.3f}\np = {final_ps[j]:.3f}', 
                transform=ax.transAxes, 
                ha='left', va='top', 
                fontsize=6,  # Optional, adjust as needed
                color='blue'  # Optional, adjust as needed
            )
            # q: make the x axis logarithmic

        plt.tight_layout()
        p_vals.append(p_vals_session); p_vals_prod.append(final_ps)
        fig.savefig(os.path.join(save_dir, f'session{session}.png'), dpi=300)
        if n_fig > 1: plt.close()

    return p_vals, p_vals_prod

def plot_inset_tuning(ax, tuning, vmin=None, vmax=None):
    cmap = plt.cm.get_cmap('jet')
    if vmin is None: vmin = np.nanmin(tuning)
    if vmax is None: vmax = np.nanmax(tuning)

    for i in range(tuning.shape[-1]):
        col = cmap((tuning[i]-vmin)/(vmax-vmin))
        ax.bar(i, tuning[i], color=col, width=1, edgecolor='none')
        ax.set_ylim(bottom=0.95*np.nanmin(tuning), top=1.05*np.nanmax(tuning))

        ax.set_xlabel('Value')
        ax.xaxis.grid(True, which='minor', color='lightgrey', linewidth=0.5)
        ax.xaxis.grid(True, which='major', linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(tuning.shape[-1]/4))
        ax.xaxis.set_minor_locator(MultipleLocator(tuning.shape[-1]/16))
        ax.set_xticklabels([])
        ax.set_xlim([-0, tuning.shape[-1]-0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

# def get_tunings(PFC, values):
#     use_sig = False
#     fold = False
#     n_p_bins = 100
#     n = 11

#     tuning_means, tuning_stds, samples = [], [], []
#     firing_by_bins_sessions = []
#     firing_rates = PFC['Data'][0]
#     for session, (value, firing_rate) in enumerate(zip(values, firing_rates)):
#         firing_rate = firing_rate[n:]
#         n_neurons = firing_rate.shape[1]; n_timepoints = firing_rate.shape[-1]

#         if use_sig: 
#             belief = logistic.cdf(value)
#             if fold: belief = 0.5 + np.abs(belief - 0.5)
#             lims = [0, 1] if not fold else [0.5, 1]
#             bin_edges = np.linspace(lims[0], lims[1], n_p_bins+1)[1:-1]  # Creates 5 edges for 4 bins
#             bins = np.digitize(belief, bin_edges)
#         else:
#             if fold: value = np.abs(value)
#             lims = [-np.max(np.abs(value)), np.max(np.abs(value))] if not fold else [0, np.max(np.abs(value))]
#             bin_edges = np.linspace(lims[0], lims[1], n_p_bins+1)[1:-1]  # Creates 5 edges for 4 bins
#             bins = np.digitize(value, bin_edges)

#         # Number of unique states
#         bins = bins.flatten()
#         num_states = n_p_bins

#         print(len(bins), firing_rate.shape)

#         sample = np.zeros(n_p_bins)
#         tuning_mean = np.zeros((n_neurons, n_timepoints, n_p_bins)); tuning_std = np.zeros((n_neurons, n_timepoints, n_p_bins))
#         firing_by_bins = []
#         for p in range(n_p_bins):
#             p_mask = bins == p
#             p_firing_rate = firing_rate[p_mask]
#             tuning_mean[:, :, p] = np.nanmean(p_firing_rate, axis=0)
#             tuning_std[:, :, p] = np.nanstd(p_firing_rate, axis=0)
#             firing_by_bins.append(p_firing_rate)
#             sample[p] = np.count_nonzero(p_mask)

#         tuning_means.append(tuning_mean); tuning_stds.append(tuning_std); samples.append(sample); firing_by_bins_sessions.append(firing_by_bins)

#         return tuning_means, tuning_stds, samples
# """