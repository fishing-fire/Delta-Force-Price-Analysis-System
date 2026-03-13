import collections
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from tqdm import tqdm

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 3))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 5))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def visual_experiment_results(trues, preds, folder_path, num_samples=8):
    """
    Generate detailed visualization for experiment results.
    trues: np.ndarray (N, L, D)
    preds: np.ndarray (N, L, D)
    folder_path: str, path to save images
    num_samples: int, number of random samples to plot
    """
    
    # Ensure directory exists
    vis_path = os.path.join(folder_path, 'visualization')
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    metrics_path = os.path.join(folder_path, 'metrics.npy')
    if os.path.exists(metrics_path):
        metrics = np.load(metrics_path)
        metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
        cell_text = [[name, f'{float(value):.6g}'] for name, value in zip(metric_names, metrics.tolist())]
        fig, ax = plt.subplots(figsize=(6.5, 2.2))
        ax.axis('off')
        table = ax.table(
            cellText=cell_text,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.4)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_path, 'metrics_table.png'), bbox_inches='tight', dpi=200)
        plt.close(fig)
        
    N, L, D = trues.shape
    
    # Select random samples
    # Use fixed seed for reproducibility if needed, but random is fine for exploration
    indices = np.random.choice(N, min(num_samples, N), replace=False)
    indices = np.sort(indices)
    
    # Plot individual samples
    for i, idx in enumerate(indices):
        plt.figure(figsize=(12, 6))
        # Plot the last feature channel
        plt.plot(trues[idx, :, -1], label='GroundTruth', linewidth=2)
        plt.plot(preds[idx, :, -1], label='Prediction', linewidth=2)
        plt.title(f'Sample {idx} - Last Feature')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(vis_path, f'sample_{idx}.png'), bbox_inches='tight')
        plt.close()
        
    # Plot a combined summary (grid view)
    if len(indices) > 1:
        cols = 2
        rows = (len(indices) + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, idx in enumerate(indices):
            if i < len(axes):
                axes[i].plot(trues[idx, :, -1], label='GroundTruth')
                axes[i].plot(preds[idx, :, -1], label='Prediction')
                axes[i].set_title(f'Sample {idx} - Last Feature')
                axes[i].legend()
                axes[i].grid(True)
        
        # Hide empty subplots
        for i in range(len(indices), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(vis_path, 'summary_grid.png'), bbox_inches='tight')
        plt.close()



def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def custom_collate(batch):
    r"""source: pytorch 1.9.0, only one modification to original code """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return custom_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: custom_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(custom_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [custom_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

class HiddenPrints:
    def __init__(self, rank):
        # 如果rank是none，那么就是单机单卡，不需要隐藏打印，将rank设置为0
        if rank is None:
            rank = 0
        self.rank = rank
    def __enter__(self):
        if self.rank == 0:
            return
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rank == 0:
            return
        sys.stdout.close()
        sys.stdout = self._original_stdout

def visual_full_test_set(trues, preds, folder_path):
    """
    Generate full visualization for all test samples and global trend.
    trues: np.ndarray (N, L, D)
    preds: np.ndarray (N, L, D)
    folder_path: str, path to save images
    """
    print(f"Generating full test set visualizations for {folder_path}...")
    
    # Create visualization directory
    vis_path = os.path.join(folder_path, 'visualization_full')
    os.makedirs(vis_path, exist_ok=True)
    
    N, L, D = trues.shape

    # Loop through all channels
    for d in range(D):
        channel_path = os.path.join(vis_path, f'channel_{d}')
        os.makedirs(channel_path, exist_ok=True)
        
        all_samples_path = os.path.join(channel_path, 'all_samples')
        os.makedirs(all_samples_path, exist_ok=True)

        # 1. Plot Global Trend (Concatenated) for channel d
        try:
            # We reconstruct the series by taking index 0 of each window, plus the rest of the last window.
            full_true = np.concatenate([trues[:-1, 0, d], trues[-1, :, d]])
            full_pred = np.concatenate([preds[:-1, 0, d], preds[-1, :, d]])

            plt.figure(figsize=(20, 6))
            plt.plot(full_true, label='GroundTruth', linewidth=1.5, alpha=0.8)
            plt.plot(full_pred, label='Prediction (One-Step)', linewidth=1.5, alpha=0.8)
            plt.title(f'Global Trend (Full Test Set) - Feature {d}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(channel_path, 'global_trend.png'), dpi=200)
            plt.close()
        except Exception as e:
            print(f"Error plotting global trend for feature {d}: {e}")

        # 2. Plot All Individual Samples for channel d
        print(f"  Generating {N} individual sample plots for Feature {d}...")
        
        # Use tqdm for progress bar
        for i in tqdm(range(N), desc=f"Plotting Samples (Feat {d})", unit="img"):
            plt.figure(figsize=(10, 5))
            plt.plot(trues[i, :, d], label='GroundTruth', linewidth=2)
            plt.plot(preds[i, :, d], label='Prediction', linewidth=2)
            plt.title(f'Sample {i} - Feature {d}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(all_samples_path, f'sample_{i}.png'), bbox_inches='tight')
            plt.close()

    print(f"  Full visualization completed in {vis_path}")
