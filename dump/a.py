class LinearModel(nn.Module):
    """
    A simple linear regression model suitable for time series forecasting.

    Parameters:
    - input_size (int): Number of input features.

    Attributes:
    - linear (nn.Linear): A linear layer that transforms input features into a single output.

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Implements the forward propagation of the model.

    Example:
    --------
    >>> model = LinearModel(input_size=10)
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
