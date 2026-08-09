import torch
import torch.nn as nn
import copy

class Encoder(nn.Module):
    """
    Encodes a history window of temperature and action pairs into a latent state representation.
    Input size: 2 * L (where L is the window length, each step containing [Temp, PWM])
    Output size: latent_dim
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    """
    Predicts the next latent state representation given the current state embedding and action.
    Input size: latent_dim (state) + 1 (action PWM)
    Output size: latent_dim (next state)
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
    def forward(self, state, action):
        # Concatenate state and action along the last dimension
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class Decoder(nn.Module):
    """
    Decodes the current physical temperature from a latent state representation (grounding).
    Input size: latent_dim
    Output size: 1 (temperature)
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, state):
        return self.net(state)

class JEPAWorldModel(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA) World Model.
    Coordinates the context encoder, target encoder, predictor, and readout decoder.
    """
    def __init__(self, window_length=10, latent_dim=8):
        super().__init__()
        self.window_length = window_length
        self.latent_dim = latent_dim
        self.input_dim = 2 * window_length  # L times [Temp, PWM]
        
        # Context Encoder (trained via gradients)
        self.context_encoder = Encoder(self.input_dim, self.latent_dim)
        
        # Target Encoder (updated via EMA of context encoder)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        # Freeze target encoder parameters
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        # Predictor (trained via gradients)
        self.predictor = Predictor(self.latent_dim)
        
        # Readout Decoder (trained via gradients to ground latent space in temperature)
        self.decoder = Decoder(self.latent_dim)

    def forward(self, context_window, action):
        """
        Runs a prediction step.
        context_window: Tensor of shape (batch, 2 * L)
        action: Tensor of shape (batch, 1) containing current PWM (normalized 0..1 or raw)
        Returns:
            predicted_next_state (batch, latent_dim)
        """
        state = self.context_encoder(context_window)
        next_state_pred = self.predictor(state, action)
        return next_state_pred

    def encode_context(self, context_window):
        """Encodes a context window using the gradient-trained encoder."""
        return self.context_encoder(context_window)

    def encode_target(self, target_window):
        """Encodes a target window using the EMA target encoder (no gradients)."""
        with torch.no_grad():
            return self.target_encoder(target_window)

    def decode_temp(self, state):
        """Decodes the temperature from a state embedding."""
        return self.decoder(state)

    def update_target_encoder(self, alpha=0.99):
        """
        Updates the target encoder parameters using an Exponential Moving Average (EMA).
        theta_target = alpha * theta_target + (1 - alpha) * theta_context
        """
        with torch.no_grad():
            for p_target, p_context in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
                p_target.data.mul_(alpha).add_(p_context.data, alpha=1 - alpha)
