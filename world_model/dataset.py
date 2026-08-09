import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def prepare_data(csv_path, window_length=10):
    """
    Reads a CSV dataset and extracts sliding windows for JEPA training.
    
    The CSV should have 'temperature' and 'pwm' columns.
    Returns:
        contexts (np.ndarray): shape (N_samples, 2 * window_length)
        actions (np.ndarray): shape (N_samples, 1) - current step action
        targets (np.ndarray): shape (N_samples, 2 * window_length) - shifted window
        target_temps (np.ndarray): shape (N_samples, 1) - physical target temp for grounding
        stats (dict): normalization statistics (mean, std) for de/serialization
    """
    df = pd.read_csv(csv_path)
    temps = df['temperature'].values
    pwms = df['pwm'].values
    
    # Calculate normalization parameters
    t_mean = float(np.mean(temps))
    t_std = float(np.std(temps))
    if t_std < 1e-4:
        t_std = 1.0
        
    stats = {
        'temp_mean': t_mean,
        'temp_std': t_std
    }
    
    # Normalize inputs
    # Temperatures: z-score
    norm_temps = (temps - t_mean) / t_std
    # PWMs: scale 0..255 to 0..1
    norm_pwms = pwms / 255.0
    
    # Construct sequence windows
    # At step t, context window contains pairs up to t (T_t, u_{t-1})
    # Target window is shifted by 1 (T_{t+1}, u_t)
    # Action is u_t (transitions from state t to t+1)
    # Target temperature is T_{t+1} (raw physical temp, for grounding)
    
    contexts = []
    actions = []
    targets = []
    target_temps = []
    
    N = len(df)
    for i in range(window_length - 1, N - 1):
        # We need at least one step in the future for target window and next temperature
        # Context window: indexes [i - window_length + 1 ... i]
        ctx_temps = norm_temps[i - window_length + 1 : i + 1]
        ctx_pwms = norm_pwms[i - window_length + 1 : i + 1]
        
        # Target window: indexes [i - window_length + 2 ... i + 1]
        targ_temps = norm_temps[i - window_length + 2 : i + 2]
        targ_pwms = norm_pwms[i - window_length + 2 : i + 2]
        
        # Interleave [T_0, u_0, T_1, u_1, ...]
        ctx = np.empty(2 * window_length, dtype=np.float32)
        ctx[0::2] = ctx_temps
        # Since u_{t} is future relative to T_{t}, in the context window we have T_t and u_{t-1}.
        # So we pad the first u index with 0.0, or use pwms shifted. Let's do:
        # ctx = [T_{t-L+1}, u_{t-L}, T_{t-L+2}, u_{t-L+1}, ..., T_t, u_{t-1}]
        # This is clean and causal!
        # For the first element, u_{t-L} is pwms[i - window_length + 1], which is the action applied
        # AFTER reading T_{t-L+1}.
        ctx[1::2] = ctx_pwms
        
        targ = np.empty(2 * window_length, dtype=np.float32)
        targ[0::2] = targ_temps
        targ[1::2] = targ_pwms
        
        act = np.array([norm_pwms[i]], dtype=np.float32)  # Action u_t
        targ_t = np.array([temps[i + 1]], dtype=np.float32) # Target temp T_{t+1} in physical unit
        
        contexts.append(ctx)
        actions.append(act)
        targets.append(targ)
        target_temps.append(targ_t)
        
    return (
        np.array(contexts),
        np.array(actions),
        np.array(targets),
        np.array(target_temps),
        stats
    )

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapping the sequence inputs.
    """
    def __init__(self, contexts, actions, targets, target_temps):
        self.contexts = torch.tensor(contexts, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.target_temps = torch.tensor(target_temps, dtype=torch.float32)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {
            'context': self.contexts[idx],
            'action': self.actions[idx],
            'target': self.targets[idx],
            'target_temp': self.target_temps[idx]
        }
