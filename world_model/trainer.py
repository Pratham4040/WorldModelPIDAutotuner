import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

class JEPATrainer:
    """
    Trainer class for the JEPA World Model.
    Minimizes the self-supervised latent prediction loss and the temperature grounding loss,
    while performing EMA target updates to prevent representation collapse.
    """
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, ema_decay=0.99, lambda_dec=1.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.ema_decay = ema_decay
        self.lambda_dec = lambda_dec
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        self.mse_loss = nn.MSELoss()

    def train_epoch(self, dataloader, device):
        self.model.train()
        epoch_jepa_loss = 0.0
        epoch_dec_loss = 0.0
        epoch_total_loss = 0.0
        
        for batch in dataloader:
            contexts = batch['context'].to(device)
            actions = batch['action'].to(device)
            targets = batch['target'].to(device)
            
            # 1. Zero gradients
            self.optimizer.zero_grad()
            
            # 2. Get embeddings
            # State embedding (from context encoder, allows gradients)
            s_t = self.model.encode_context(contexts)
            
            # Target embedding (from target encoder, frozen, no gradients)
            s_y_t1 = self.model.encode_target(targets)
            
            # 3. Predict next embedding
            s_t1_pred = self.model.predictor(s_t, actions)
            
            # 4. Compute JEPA dynamics loss
            loss_jepa = self.mse_loss(s_t1_pred, s_y_t1)
            
            # 5. Compute temperature grounding loss
            # Predict normalized temperatures
            t_norm_t_pred = self.model.decode_temp(s_t)
            t_norm_t1_pred = self.model.decode_temp(s_t1_pred)
            
            # Extract true normalized temperatures from the windows (at index -2, since index -1 is PWM)
            t_norm_t_true = contexts[:, -2].unsqueeze(-1)
            t_norm_t1_true = targets[:, -2].unsqueeze(-1)
            
            loss_dec = self.mse_loss(t_norm_t_pred, t_norm_t_true) + self.mse_loss(t_norm_t1_pred, t_norm_t1_true)
            
            # 6. Combined loss
            loss_total = loss_jepa + self.lambda_dec * loss_dec
            
            # 7. Backward and optimize
            loss_total.backward()
            self.optimizer.step()
            
            # 8. Update target encoder weights (EMA)
            self.model.update_target_encoder(self.ema_decay)
            
            # Log
            epoch_jepa_loss += loss_jepa.item() * contexts.size(0)
            epoch_dec_loss += loss_dec.item() * contexts.size(0)
            epoch_total_loss += loss_total.item() * contexts.size(0)
            
        dataset_size = len(dataloader.dataset)
        return {
            'jepa_loss': epoch_jepa_loss / dataset_size,
            'decoder_loss': epoch_dec_loss / dataset_size,
            'total_loss': epoch_total_loss / dataset_size
        }

    def evaluate(self, dataloader, device):
        self.model.eval()
        epoch_jepa_loss = 0.0
        epoch_dec_loss = 0.0
        epoch_total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                contexts = batch['context'].to(device)
                actions = batch['action'].to(device)
                targets = batch['target'].to(device)
                
                s_t = self.model.encode_context(contexts)
                s_y_t1 = self.model.encode_target(targets)
                s_t1_pred = self.model.predictor(s_t, actions)
                
                loss_jepa = self.mse_loss(s_t1_pred, s_y_t1)
                
                t_norm_t_pred = self.model.decode_temp(s_t)
                t_norm_t1_pred = self.model.decode_temp(s_t1_pred)
                
                t_norm_t_true = contexts[:, -2].unsqueeze(-1)
                t_norm_t1_true = targets[:, -2].unsqueeze(-1)
                
                loss_dec = self.mse_loss(t_norm_t_pred, t_norm_t_true) + self.mse_loss(t_norm_t1_pred, t_norm_t1_true)
                loss_total = loss_jepa + self.lambda_dec * loss_dec
                
                epoch_jepa_loss += loss_jepa.item() * contexts.size(0)
                epoch_dec_loss += loss_dec.item() * contexts.size(0)
                epoch_total_loss += loss_total.item() * contexts.size(0)
                
        dataset_size = len(dataloader.dataset)
        return {
            'jepa_loss': epoch_jepa_loss / dataset_size,
            'decoder_loss': epoch_dec_loss / dataset_size,
            'total_loss': epoch_total_loss / dataset_size
        }

    def train(self, train_dataset, val_dataset, epochs=20, batch_size=32, device='cpu'):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        history = {
            'train_jepa': [], 'train_dec': [], 'train_total': [],
            'val_jepa': [], 'val_dec': [], 'val_total': []
        }
        
        print(f"Starting JEPA training on {device} for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, device)
            val_metrics = self.evaluate(val_loader, device)
            
            history['train_jepa'].append(train_metrics['jepa_loss'])
            history['train_dec'].append(train_metrics['decoder_loss'])
            history['train_total'].append(train_metrics['total_loss'])
            history['val_jepa'].append(val_metrics['jepa_loss'])
            history['val_dec'].append(val_metrics['decoder_loss'])
            history['val_total'].append(val_metrics['total_loss'])
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_metrics['total_loss']:.4f} (JEPA: {train_metrics['jepa_loss']:.4f}, Dec: {train_metrics['decoder_loss']:.4f}) | "
                  f"Val Loss: {val_metrics['total_loss']:.4f} (JEPA: {val_metrics['jepa_loss']:.4f}, Dec: {val_metrics['decoder_loss']:.4f})")
                  
        return history
