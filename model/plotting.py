import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from config import Conf
import os
from tqdm import tqdm
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_trials(inputs, ground_truth, targets, choices, p_A, b=0, trials=200, sf=1.1, fname='panel.png'):
    t = Conf.trial_len * trials

    fig, axes = plt.subplots(1,6, figsize=(sf*4, sf*t//20), dpi=200)
    axes[0].imshow(inputs[b, :t, :], vmin=0, vmax=1, aspect='auto')
    axes[0].set_title('Input', fontsize=8)
    axes[0].set_ylabel('Trial #')
    axes[1].imshow(ground_truth[b, :t, :], vmin=0, vmax=1, aspect='auto')
    axes[1].set_title('Ground truth', fontsize=8)
    axes[2].imshow(targets[b, :t, :], vmin=0, vmax=1, aspect='auto')
    axes[2].set_title('Target', fontsize=8)
    axes[3].imshow(choices[b, :t, :], vmin=0, vmax=1, aspect='auto')
    axes[3].set_title('Softmax', fontsize=8)

    # # Draw white lines every 4 rows
    # for i in range(1, choices[b, :t, :].shape[0] // 4):
    #     axes[0].axhline(y=i * 4 - 0.5, color='white', linewidth=2)
    #     axes[1].axhline(y=i * 4 - 0.5, color='white', linewidth=2)
    #     axes[2].axhline(y=i * 4 - 0.5, color='white', linewidth=2)

    # Set yticks at every 4th row and enable gridlines only for those ticks
    for i, ax in enumerate(axes):
        if i == len(axes)-1:
            start, gap, rows = 0.5, 1, trials
            ax.set_xticks(np.arange(0.5, Conf.trial_len, 1))
            ax.xaxis.grid(True, linestyle='-', color='white', linewidth=1)
        else:
            start, gap, rows = Conf.trial_len-0.5, Conf.trial_len, t
            ax.xaxis.grid(False)
        # Generate ytick locations, starting from 3.5 (since we want lines between blocks of 4)
        yticks = np.arange(start, rows, gap)
        ax.set_yticks(yticks)

        # Enable grid only for y-axis
        ax.yaxis.grid(True, linestyle='-', color='white', linewidth=2)

        # Optional: Hide x-axis grid lines
    # Get the column indices of the largest value in each row of choices
    max_choices = np.argmax(choices[b, :t, :], axis=1)
    max_targets = np.argmax(targets[b, :t, :], axis=1)

    # Iterate through each row and draw a red box around the cell with the largest value
    for i, (choice, target) in enumerate(zip(max_choices, max_targets)):
        c = 'green' if choice==target else 'red'
        rect = patches.Rectangle((choice-0.5, i-0.5), 1, 1, linewidth=1.5, edgecolor=c, facecolor='none')
        axes[3].add_patch(rect)

    correct = (np.argmax(choices, axis=-1) == np.argmax(targets, axis=-1)).reshape((choices.shape[0], choices.shape[1]//Conf.trial_len, Conf.trial_len))
    # axes[3].imshow(correct[b, :trials, :], aspect='auto')
    axes[4].set_title('Correct', fontsize=8)
    # Create a color matrix with the same shape as your correct/incorrect matrix
    # Initialize it fully transparent
    color_matrix = np.zeros((*correct[b, :trials, :].shape, 4))

    # Set red color with alpha=0.5 for cells with 0
    color_matrix[correct[b, :trials, :] == 0] = [1, 0, 0, 0.3]  # Red with alpha 0.5

    # Set green color with alpha=0.5 for cells with 1
    color_matrix[correct[b, :trials, :] == 1] = [0, 1, 0, 0.3]  # Green with alpha 0.5

    # Now identify rows where all values are 1 and set alpha=1.0 for those cells
    all_correct_rows = np.all(correct[b, :trials, :] == 1, axis=1)
    color_matrix[all_correct_rows, :, 3] = 1.0  # Set alpha to 1.0 for these rows

    # Display the color matrix using imshow
    axes[4].imshow(color_matrix, aspect='auto')
    axes[4].set_xticks(np.arange(0.5, Conf.trial_len-0.5, 1))
    axes[4].set_yticks(np.arange(0.5, trials-0.5, 1))
    axes[4].grid(True, linestyle='-', color='white', linewidth=1)
    # q: how do I turn on grid on this axis
    # a: 

    for ax in axes:
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    axes[5].plot(p_A[b, :trials], 0.5 + np.arange(trials), marker='o', markersize=3, color='b')
    axes[5].set_ylim([0, trials])
    axes[5].set_xlim([0, 1])
    axes[5].invert_yaxis()
    axes[5].axvline(x=0.5, color='red', linestyle='--', linewidth=1)
    axes[5].set_title('p(A)', fontsize=8)
    # q: how do I plot this vertically, so that the x axis and y axis are swapped?
    # 

    proportion_correct = 100 * np.sum(correct, axis=(0,1)) / (correct.shape[0]*correct.shape[1])
    proportion_correct = ', '.join([f'{p:.1f}' for p in proportion_correct])
    print(proportion_correct)
    # Set the suptitle of the figure with the calculated proportion
    fig.suptitle('Accs: ' + proportion_correct)

    plt.tight_layout(rect=[0, 0, 1, 0.985]) # Adjust layout to make space for suptitle
    plt.show()

    fig.savefig(os.path.join(Conf.save_dir, fname))


# def plot_average_layout_hists(p_A, inputs_arg_trial, hidden_trial, anomalous_batches, neur_ids=None, n_p_bins=50):
#     n_neurons = 256
#     neur_ids = neur_ids if neur_ids is not None else range(np.minimum(n_neurons, hidden_trial.shape[-1]))
#     polar = True
#     separate_figures = True
#     overwrite = True
#     init_step, a_step, b_step, choice_step = 2, 3, 4, 4
#     # load_dir = '../run_data_1_128_l2_1e6_l2_3e4'

#     bin_edges = np.linspace(1, 0, n_p_bins+1)  # Creates 5 edges for 4 bins
#     bins = np.digitize(p_A, bin_edges) - 1

#     cmap = cm.get_cmap('cool', n_p_bins) if n_p_bins > 2 else cm.get_cmap('bwr', n_p_bins) # Get a colormap object

#     if not separate_figures:
#         raise NotImplementedError()
#     else:
#         n_col = Conf.port_dim
#         n_row = 3
#         # Adjusting figure size based on the number of batches and neurons
#         fig_width = 16  # Adjust width as needed
#         fig_height = 9  # Adjust height as needed
#         save_dir = os.path.join(Conf.save_dir, 'separate_figs', 'polar' if polar else 'planar', 'mean', 'hist_griddy')
#         print(save_dir)
#         if not os.path.exists(save_dir): os.makedirs(save_dir)

#     theta = np.linspace(0, 2*np.pi, Conf.trial_len, endpoint=False)
#     if polar:
#         theta = np.append(theta, theta[0])

#     for k in tqdm(neur_ids):
#         # vmax = np.minimum(Conf.threshold, np.mean(hiddens[:, 5*Conf.trial_len:, k] + 2 * np.std(hiddens[:, 5*Conf.trial_len:, k])))
#         vmax = np.percentile(hidden_trial[:,:,:,k], 99)
#         # print(vmax, vmax_, np.where(hiddens[:, 5*Conf.trial_len:, k] == vmax_))

#         if separate_figures:
#             fpath = os.path.join(save_dir, str(k) + '.png')
#             if not overwrite and os.path.exists(fpath):
#                 print(f'file {fpath} already exists')
#                 continue
#             fig, axes = plt.subplots(n_row, n_col, figsize=(fig_width, fig_height), layout='constrained')
#             fig.suptitle(f'Neuron {k}', fontsize=14)
#         else:
#             raise NotImplementedError()

#         for port_id in range(Conf.port_dim):
#             for row, step in enumerate([init_step, a_step, b_step]):
#                 ax = axes[row, port_id]
#                 ax.xaxis.set_visible(False)
#                 # ax.yaxis.set_visible(False)
#                 ax.set_yticks([])
#                 ax.set_xticks([])
#                 ax.spines['right'].set_visible(False)
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['bottom'].set_visible(False)
#                 ax.spines['left'].set_visible(False)
#                 if port_id == 0:
#                     ax.set_ylabel('firing rate')
#                 # if row == n_row - 1:
#                 #     ax.set_xlabel('p')

#                 title = (str(port_id) if row == 0 else 'x') + ', ' + (str(port_id) if row == 1 else 'x') + '/' + (str(port_id) if row == 2 else 'x')
#                 ax.set_title(title, fontsize=12, fontweight='bold')

#                 step_type_trial_mask = inputs_arg_trial[:, :, step] == port_id
#                 # mask = step_type_trial_mask
#                 # port_trial_mask = choices_arg_trial[:, :, init_step if step==init_step else choice_step] == port_id
#                 anomaly_mask = anomalous_batches[:, np.newaxis] * np.ones_like(step_type_trial_mask, dtype=bool)
#                 # print(step_type_trial_mask.shape, anomaly_mask.shape)
#                 mask = np.logical_and(np.logical_not(anomaly_mask), step_type_trial_mask)

#                 bin_heights = np.zeros((Conf.trial_len, n_p_bins))
#                 for bin_id in range(n_p_bins):
#                     p_bin_mask = bins == bin_id

#                     if np.any(p_bin_mask):
#                         trial_mask = np.logical_and(mask, p_bin_mask)

#                         trial_idxs = np.where(trial_mask)
#                         # print(port_id, ['init', 'a', 'b'][row], bin_id, trial_idxs[0].shape)

#                         h = hidden_trial[trial_idxs[0], trial_idxs[1], :, k]
#                         h_mean = np.mean(h, axis=0)
#                         # print(h_mean, np.std(h, axis=0))
#                         bin_heights[:, bin_id] = np.roll(h_mean, shift=-2)
#                 # q: how do I make 5 axes within main axis?

#                 # Parameters for inner axes
#                 inner_ax_width = "100%"  # Example width
#                 inner_ax_height = "10%" # Example height

#                 y_shift = (1 - 0.3)/Conf.trial_len
#                 # Create and plot in inner axes
#                 for i in range(Conf.trial_len):
#                     inner_ax = inset_axes(ax, width=inner_ax_width, height=inner_ax_height, 
#                                         loc=2, bbox_to_anchor=(0.1, -0.1-y_shift*i, 0.8, 1),
#                                         bbox_transform=ax.transAxes, borderpad=0)

#                     # Example plotting in inner_ax
#                     # inner_ax.plot([0, 1], [0, 1], 'r-') 
#                     x = np.flip(bin_edges)
#                     # inner_ax.hist(x[:-1], x, weights=bin_heights[i,:], color=cmap(np.arange(n_p_bins)))
#                     # Now plot each bar with its respective color
#                     centres = (x[:-1] + x[1:]) / 2
#                     for j, height in enumerate(bin_heights[i, :]):
#                         inner_ax.bar(centres[j], height, color=cmap(j), width=1/n_p_bins)

#                     inner_ax.set_ylim([0, vmax])
#                     inner_ax.set_xlim([0, 1])
                    
#                     # Turn off tick labels
#                     if port_id == 0:
#                         inner_ax.set_ylabel(['init', 'delay', 'choice', 'reward', 'ITI'][i], fontsize=6)
#                     inner_ax.set_yticks([])
#                     inner_ax.spines['right'].set_visible(False)
#                     inner_ax.spines['top'].set_visible(False)
#                     # inner_ax.spines['left'].set_visible(False)
#                     if (i == Conf.trial_len - 1):# and (row == n_row - 1):
#                         # inner_ax.set_xticks([0, 1], ['A', 'B'])
#                         inner_ax.set_xticks([0.5], ['p(B)'])
#                         inner_ax.tick_params(axis='x', which='both', length=0, pad=6)
#                         # inner_ax.set_xlabel('p(B)', fontsize=6)
#                     else:
#                         inner_ax.set_xticks([])

#                     # inner_ax.set_title(f'step {i}', fontsize=8)

#             # Parameters for the colorbar axis (left, bottom, width, height)
#             cbar_ax = fig.add_axes([0.947, 0.987, 0.05, 0.01])  # Adjust these values as needed
#             cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cbar_ax, orientation='horizontal')
#             # Set the colorbar ticks and labels
#             cbar.set_ticks([0, 1])
#             cbar.set_ticklabels(['A', 'B'])
#             cbar.ax.tick_params(direction='in', pad=3, labelsize=10)

#             fig.text(0.002, 0.998, 'init, a/b', fontsize=12, fontweight='bold', ha='left', va='top')   

#             fig.savefig(fpath)              


def plot_average_layout_hists(means, neur_ids=None, overwrite=False):
    neur_ids = neur_ids if neur_ids is not None else np.arange(means.shape[-2])
    separate_figures = True
    # init_step, a_step, b_step = 2, 3, 4
    # load_dir = '../run_data_1_128_l2_1e6_l2_3e4'

    n_p_bins = means.shape[2]
    bin_edges = np.linspace(1, 0, n_p_bins+1)  # Creates 5 edges for 4 bins

    cmap = cm.get_cmap('cool', n_p_bins) if n_p_bins > 2 else cm.get_cmap('bwr', n_p_bins) # Get a colormap object

    if not separate_figures:
        raise NotImplementedError()
    else:
        n_col = Conf.port_dim
        n_row = 3
        # Adjusting figure size based on the number of batches and neurons
        fig_width = 16  # Adjust width as needed
        fig_height = 9  # Adjust height as needed
        save_dir = os.path.join(Conf.save_dir, 'separate_figs', 'polar', 'mean', f'hist_griddy_{n_p_bins}')
        print(save_dir)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

    theta = np.linspace(0, 2*np.pi, Conf.trial_len, endpoint=False)

    for k in tqdm(neur_ids):
        # vmax = np.minimum(Conf.threshold, np.mean(hiddens[:, 5*Conf.trial_len:, k] + 2 * np.std(hiddens[:, 5*Conf.trial_len:, k])))
        vmax = np.percentile(means[:,:,:,k,:], 99.9)
        # print(vmax, vmax_, np.where(hiddens[:, 5*Conf.trial_len:, k] == vmax_))

        if separate_figures:
            fpath = os.path.join(save_dir, str(k) + '.png')
            print(fpath)
            if not overwrite and os.path.exists(fpath):
                print(f'file {fpath} already exists')
                continue
            fig, axes = plt.subplots(n_row, n_col, figsize=(fig_width, fig_height), layout='constrained')
            fig.suptitle(f'Neuron {k}', fontsize=14)
        else:
            raise NotImplementedError()

        for port_id in range(Conf.port_dim):
            for row, step in enumerate([Conf.init_step, Conf.a_step, Conf.b_step]):
                ax = axes[row, port_id]
                ax.xaxis.set_visible(False)
                # ax.yaxis.set_visible(False)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                if port_id == 0:
                    ax.set_ylabel('firing rate')
                # if row == n_row - 1:
                #     ax.set_xlabel('p')

                title = (str(port_id) if row == 0 else 'x') + ', ' + (str(port_id) if row == 1 else 'x') + '/' + (str(port_id) if row == 2 else 'x')
                ax.set_title(title, fontsize=12, fontweight='bold')

                bin_heights = np.zeros((Conf.trial_len, n_p_bins))
                for bin_id in range(n_p_bins):
                    h_mean = means[row, port_id, bin_id, k, :]
                    bin_heights[:, bin_id] = np.roll(h_mean, shift=-Conf.init_step)
                # q: how do I make 5 axes within main axis?

                # Parameters for inner axes
                inner_ax_width = "100%"  # Example width
                inner_ax_height = "5%" # Example height

                y_shift = (1 - 0.3)/Conf.trial_len
                # Create and plot in inner axes
                for i in range(Conf.trial_len):
                    inner_ax = inset_axes(ax, width=inner_ax_width, height=inner_ax_height, 
                                        loc=2, bbox_to_anchor=(0.1, -0.1-y_shift*i, 0.8, 1),
                                        bbox_transform=ax.transAxes, borderpad=0)

                    # Example plotting in inner_ax
                    # inner_ax.plot([0, 1], [0, 1], 'r-') 
                    x = np.flip(bin_edges)
                    # inner_ax.hist(x[:-1], x, weights=bin_heights[i,:], color=cmap(np.arange(n_p_bins)))
                    # Now plot each bar with its respective color
                    centres = (x[:-1] + x[1:]) / 2
                    for j, height in enumerate(bin_heights[i, :]):
                        inner_ax.bar(centres[j], height, color=cmap(j), width=1/n_p_bins)

                    inner_ax.set_ylim([0, vmax])
                    inner_ax.set_xlim([0, 1])
                    
                    # Turn off tick labels
                    if port_id == 0:
                        y_labels = ['0\n0' for _ in range(Conf.trial_len)]
                        y_labels[Conf.init_step] = '0\ninit'
                        y_labels[Conf.init_choice_step] = 'init\n0'
                        y_labels[Conf.a_step] = '0\na'
                        y_labels[Conf.b_step] = '0\nb'
                        y_labels[Conf.ab_choice_step] = 'ch\n0'
                        y_labels[Conf.r_step] = '0\nrew'
                        inner_ax.set_ylabel(np.roll(y_labels, shift=-Conf.init_step)[i], fontsize=6)
                    inner_ax.set_yticks([])
                    inner_ax.spines['right'].set_visible(False)
                    inner_ax.spines['top'].set_visible(False)
                    # inner_ax.spines['left'].set_visible(False)
                    if (i == Conf.trial_len - 1):# and (row == n_row - 1):
                        # inner_ax.set_xticks([0, 1], ['A', 'B'])
                        n_ticks = 10
                        ticks = np.linspace(0, 1, n_ticks+1, endpoint=True)
                        assert 0.5 in ticks
                        inner_ax.set_xticks(ticks, ['' if tick!=0.5 else 'p(B)' for tick in ticks])
                        inner_ax.tick_params(axis='x', which='both', length=2, pad=8)
                        # inner_ax.set_xlabel('p(B)', fontsize=6)
                    else:
                        inner_ax.set_xticks([])

                    # inner_ax.set_title(f'step {i}', fontsize=8)

            # Parameters for the colorbar axis (left, bottom, width, height)
            cbar_ax = fig.add_axes([0.947, 0.987, 0.05, 0.01])  # Adjust these values as needed
            cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cbar_ax, orientation='horizontal')
            # Set the colorbar ticks and labels
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['A', 'B'])
            cbar.ax.tick_params(direction='in', pad=3, labelsize=10)

            fig.text(0.002, 0.998, 'init, a/b', fontsize=12, fontweight='bold', ha='left', va='top')   

            fig.savefig(fpath)      
            plt.close()        
                     
                    
def get_means(p_A, inputs_arg_trial, hidden_trial, anomalous_batches, n_p_bins=50): 
    n_neurons = hidden_trial.shape[-1]
    bin_edges = np.linspace(1, 0, n_p_bins+1)  # Creates 5 edges for 4 bins
    bins = np.digitize(p_A, bin_edges) - 1 
    conds = [Conf.init_step, Conf.a_step, Conf.b_step]
    h_mean_all_layouts = np.zeros((len(conds), Conf.port_dim, n_p_bins, n_neurons, Conf.trial_len))
    # for k in neur_ids:
    #     vmax = np.percentile(hidden_trial[:,:,:,k], 99)

    for port_id in range(Conf.port_dim):
        for row, step in enumerate(conds):
            step_type_trial_mask = inputs_arg_trial[:, :, step] == port_id
            anomaly_mask = anomalous_batches[:, np.newaxis] * np.ones_like(step_type_trial_mask, dtype=bool)
            # print(step_type_trial_mask.shape, anomaly_mask.shape)
            mask = np.logical_and(np.logical_not(anomaly_mask), step_type_trial_mask)

            for bin_id in range(n_p_bins):
                p_bin_mask = bins == bin_id

                if np.any(p_bin_mask):
                    trial_mask = np.logical_and(mask, p_bin_mask)

                    trial_idxs = np.where(trial_mask)
                    # print(port_id, ['init', 'a', 'b'][row], bin_id, trial_idxs[0].shape)

                    h = hidden_trial[trial_idxs[0], trial_idxs[1], :, :]
                    h_mean = np.mean(h, axis=0)

                    h_mean_all_layouts[row, port_id, bin_id, :, :] = np.transpose(h_mean, (1,0))

    p_A_bin_counts = np.bincount(bins.flatten(), minlength=n_p_bins)

    return h_mean_all_layouts, p_A_bin_counts
