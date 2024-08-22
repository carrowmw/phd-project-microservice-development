# phd_package/pipeline/definitions/model_definitions.py

import torch.nn as nn
from torch.nn.utils import weight_norm


class LinearModel(nn.Module):
    """
    A simple linear regression model suitable for time series forecasting.

    Parameters:
    - feature_dim (int): Number of input features.

    Attributes:
    - linear (nn.Linear): A linear layer that transforms input features into a single output.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

    Example:
    --------
    >>> model = LinearModel(feature_dim=10)
    >>> input_data = torch.randn(32, 10)  # Batch of 32, each with 10 features
    >>> output = model(input_data)

    Notes:
    ------
    - The forward method can process both 2D (batch_size, num_features) and
      3D (batch_size, sequence_len, num_features) input tensors. If the input is 3D,
      it gets reshaped to 2D.
    """

    def __init__(self, feature_dim, output_dim=1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(feature_dim, output_dim)

    def forward(self, x):

        # If x is 3D (batch_size, sequence_len, num_features), we might need to reshape it
        # x = x.reshape(x.size(0), -1)

        # Pass the input tensor through the linear layer
        x = self.linear(x)

        # Select the last output of the sequence as the final prediction
        return x[:, -1, :]


class SimpleRNN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = self.fc(hidden[-1])
        return out


class TransformerModel(nn.Module):
    def __init__(self, feature_dim, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.relu = nn.ReLU()
        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out


class TCN(nn.Module):
    def __init__(self, feature_dim, output_dim, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            *[
                TemporalBlock(
                    num_channels[i],
                    num_channels[i + 1],
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=(kernel_size - 1),
                )
                for i in range(len(num_channels) - 1)
            ]
        )
        self.linear = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        y1 = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.linear(y1[:, -1])


class Seq2Seq(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super(Seq2Seq, self).__init__()
        self.encoder = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        decoder_output, _ = self.decoder(hidden.transpose(0, 1), (hidden, cell))
        output = self.fc(decoder_output)
        return output


# class LSTMModel(nn.Module):
#     """
#     LSTM-based model designed for time series forecasting. Suitable for both univariate and multivariate time series.

#     Parameters:
#     - feature_dim (int): Number of expected features in the input `x`.
#     - hidden_dim (int, optional): Number of features in the hidden state. Default: 50.
#     - output_dim (int, optional): Number of features in the output. Default: 1.
#     - num_layers (int, optional): Number of recurrent layers. Default: 1.

#     Attributes:
#     - lstm (nn.LSTM): LSTM layer.
#     - linear (nn.Linear): Linear layer to produce the final output.

#     Methods:
#     - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

#     Example:
#     --------
#     >>> model = LSTMModel(feature_dim=10)
#     >>> input_data = torch.randn(32, 7, 10)  # Batch of 32, sequence length of 7, each with 10 features
#     >>> output = model(input_data)
#     """

#     def __init__(self, feature_dim, hidden_dim=64, output_dim=1, num_layers=1):
#         super().__init__()
#         self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         """
#         Forward propagation method for the LSTM model.

#         Args:
#         - x (torch.Tensor): Input tensor with sequences. Expected shape: [batch_size, sequence_length, feature_dim].

#         Returns:
#         - torch.Tensor: Output tensor with predictions. Shape: [batch_size, output_dim].
#         """
#         x, _ = self.lstm(x)
#         x = self.linear(x)
#         return x[:, -1, :]  # Selecting the last output of the sequence


class LSTMModel(nn.Module):
    """
    LSTM-based model designed for time series forecasting.
    Suitable for both univariate and multivariate time series.

    Parameters:
    - feature_dim (int): Number of expected features in the input `x`.
    - hidden_dim (int, optional): Number of features in the hidden state. Default: 256.
    - output_dim (int, optional): Number of features in the output. Default: 1.
    - num_layers (int, optional): Number of recurrent layers. Default: 3.
    - dropout (float, optional): Dropout probability. Default: 0.2.

    Attributes:
    - lstm (nn.LSTM): LSTM layer.
    - dropout (nn.Dropout): Dropout layer.
    - linear (nn.Linear): Linear layer to produce the final output.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

    Example:
    --------
    >>> model = LSTMModel(feature_dim=10)
    >>> input_data = torch.randn(32, 7, 10)  # Batch of 32, sequence length of 7, each with 10 features
    >>> output = model(input_data)
    """

    def __init__(
        self, feature_dim, hidden_dim=256, output_dim=1, num_layers=3, dropout=0.2
    ):
        super().__init__()

        # Create a stacked and bidirectional LSTM with the specified parameters
        self.lstm = nn.LSTM(
            feature_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Create a dropout layer with the specified dropout probability
        self.dropout = nn.Dropout(dropout)

        # Create a linear layer to map the LSTM output to the desired output dimension
        # Multiply hidden_dim by 2 to account for the bidirectional LSTM
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        Forward propagation method for the LSTM model.

        Args:
        - x (torch.Tensor): Input tensor with sequences. Expected shape: [batch_size, sequence_length, feature_dim].

        Returns:
        - torch.Tensor: Output tensor with predictions. Shape: [batch_size, output_dim].
        """
        # Pass the input through the LSTM layer
        x, _ = self.lstm(x)

        # Apply dropout regularization to the LSTM output
        x = self.dropout(x)

        # Pass the LSTM output through the linear layer to get the final predictions
        x = self.linear(x)

        # Select the last output of the sequence as the final prediction
        return x[:, -1, :]


class GRUModel(nn.Module):
    """
    GRU-based model designed for time series forecasting.
    Suitable for both univariate and multivariate time series.

    Parameters:
    - feature_dim (int): Number of expected features in the input `x`.
    - hidden_dim (int, optional): Number of features in the hidden state. Default: 256.
    - output_dim (int, optional): Number of features in the output. Default: 1.
    - num_layers (int, optional): Number of recurrent layers. Default: 3.
    - dropout (float, optional): Dropout probability. Default: 0.2.

    Attributes:
    - gru (nn.GRU): GRU layer.
    - dropout (nn.Dropout): Dropout layer.
    - linear (nn.Linear): Linear layer to produce the final output.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

    Example:
    --------
    >>> model = GRUModel(feature_dim=10)
    >>> input_data = torch.randn(32, 7, 10)  # Batch of 32, sequence length of 7, each with 10 features
    >>> output = model(input_data)
    """

    def __init__(
        self, feature_dim, hidden_dim=256, output_dim=1, num_layers=3, dropout=0.2
    ):
        super().__init__()

        # Create a stacked and bidirectional GRU with the specified parameters
        self.gru = nn.GRU(
            feature_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        # Create a dropout layer with the specified dropout probability
        self.dropout = nn.Dropout(dropout)

        # Create a linear layer to map the GRU output to the desired output dimension
        # Multiply hidden_dim by 2 to account for the bidirectional GRU
        self.linear = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """
        Forward propagation method for the GRU model.

        Args:
        - x (torch.Tensor): Input tensor with sequences. Expected shape: [batch_size, sequence_length, feature_dim].

        Returns:
        - torch.Tensor: Output tensor with predictions. Shape: [batch_size, output_dim].
        """
        # Pass the input through the GRU layer
        x, _ = self.gru(x)

        # Apply dropout regularization to the GRU output
        x = self.dropout(x)

        # Pass the GRU output through the linear layer to get the final predictions
        x = self.linear(x)

        # Select the last output of the sequence as the final prediction
        return x[:, -1, :]


# class LSTMAutoencoderModel(nn.Module):
#     """
#     LSTM-based autoencoder model designed for time series data.
#     Suitable for both univariate and multivariate time series.

#     Parameters:
#     - input_dim (int): Number of expected features in the input `x`.
#     - hidden_dim (int, optional): Number of features in the hidden state. Default: 64.
#     - num_layers (int, optional): Number of recurrent layers in the encoder and decoder. Default: 1.

#     Attributes:
#     - encoder (nn.LSTM): LSTM encoder layer.
#     - decoder (nn.LSTM): LSTM decoder layer.
#     - fc (nn.Linear): Linear layer to produce the final reconstructed output.

#     Methods:
#     - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

#     Example:
#     --------
#     >>> model = LSTMAutoencoder(input_dim=10)
#     >>> input_data = torch.randn(32, 7, 10)  # Batch of 32, sequence length of 7, each with 10 features
#     >>> reconstructed_data = model(input_data)
#     """

#     def __init__(self, input_dim, hidden_dim=64, num_layers=1):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers

#         self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
#         self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, input_dim)

#     def forward(self, x):
#         """
#         Forward propagation method for the LSTM Autoencoder model.

#         Args:
#         - x (torch.Tensor): Input tensor with sequences. Expected shape: [batch_size, sequence_length, input_dim].

#         Returns:
#         - torch.Tensor: Reconstructed tensor with the same shape as the input.
#         """
#         # Encode the input sequences
#         _, (hidden, cell) = self.encoder(x)

#         # Initialize the decoder input with zeros
#         decoder_input = torch.zeros(x.size(0), 1, self.hidden_dim).to(x.device)

#         # Initialize the decoder hidden state with the encoder's final hidden state
#         decoder_hidden = hidden

#         # Initialize the reconstructed sequence tensor
#         reconstructed_seq = []

#         # Iterate over the sequence length and generate reconstructed sequence step by step
#         for _ in range(x.size(1)):
#             output, (decoder_hidden, _) = self.decoder(
#                 decoder_input, (decoder_hidden, cell)
#             )
#             reconstructed_frame = self.fc(output)
#             reconstructed_seq.append(reconstructed_frame)
#             decoder_input = output

#         # Concatenate the reconstructed frames along the sequence dimension
#         reconstructed_seq = torch.cat(reconstructed_seq, dim=1)

#         return reconstructed_seq
