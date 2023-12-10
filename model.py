# Define the Generator, Discriminator, and Inference Network as subclasses of nn.Module
import torch
import torch.nn as nn

class Generator(nn.Module):
    """Generator function.

    Args:
      - x: features
      - t: treatments
      - y: observed labels

    Returns:
      - G_logit: estimated potential outcomes
    """
    def __init__(self, input_dim, h_dim, flag_dropout):
        super(Generator, self).__init__()

        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim + 2, h_dim) # +2 for t and y
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.dp2_1 = nn.Dropout(p=0.2)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.dp2_2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.dp2 = nn.Dropout(p=0.2)

        # if t = 0, train fc31, 32
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)

        # if t = 1, train fc41, 42
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc2_1.weight)
        nn.init.constant_(self.fc2_1.bias, 0)
        nn.init.xavier_uniform_(self.fc2_2.weight)
        nn.init.constant_(self.fc2_2.bias, 0)
        nn.init.xavier_uniform_(self.fc31.weight)
        nn.init.constant_(self.fc31.bias, 0)
        nn.init.xavier_uniform_(self.fc32.weight)
        nn.init.constant_(self.fc32.bias, 0)
        nn.init.xavier_uniform_(self.fc41.weight)
        nn.init.constant_(self.fc41.bias, 0)
        nn.init.xavier_uniform_(self.fc42.weight)
        nn.init.constant_(self.fc42.bias, 0)

    def forward(self, x, t, y):
        inputs = torch.cat([x, t, y], dim=1)
        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            h2_1 = self.dp2_1(torch.relu(self.fc2_1(h1)))
            h2_2 = self.dp2_2(torch.relu(self.fc2_2(h2_1)))
            h2 = self.dp2(torch.relu(self.fc2(h2_2)))
        else:
            h1 = torch.relu(self.fc1(inputs))
            h2_1 = torch.relu(self.fc2_1(h1))
            h2_2 = torch.relu(self.fc2_2(h2_1))
            h2 = torch.relu(self.fc2(h2_2))
        h31 = torch.relu(self.fc31(h2))
        logit1 = self.fc32(h31)
        y_hat_1 = torch.nn.Sigmoid()(logit1)
        h41 = torch.relu(self.fc41(h2))
        logit2 = self.fc42(h41)
        y_hat_2 = torch.nn.Sigmoid()(logit2)
        return torch.cat([y_hat_1, y_hat_2], dim=1)

class GeneratorDeep(nn.Module):
    """Generator function.

    Args:
      - x: features
      - t: treatments
      - y: observed labels

    Returns:
      - G_logit: estimated potential outcomes
    """
    def __init__(self, input_dim, h_dim, flag_dropout, k):
        super(GeneratorDeep, self).__init__()

        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim + 2, h_dim) # +2 for t and y
        self.dp1 = nn.Dropout(p=0.2)

        self.layers = nn.ModuleList()
        for _ in range(k-3):
            layer = nn.Linear(h_dim, h_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.layers.append(layer)
            dp = nn.Dropout(p=0.2)
            self.layers.append(dp)

        # if t = 0, train fc31, 32
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)

        # if t = 1, train fc41, 42
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc31.weight)
        nn.init.constant_(self.fc31.bias, 0)
        nn.init.xavier_uniform_(self.fc32.weight)
        nn.init.constant_(self.fc32.bias, 0)
        nn.init.xavier_uniform_(self.fc41.weight)
        nn.init.constant_(self.fc41.bias, 0)
        nn.init.xavier_uniform_(self.fc42.weight)
        nn.init.constant_(self.fc42.bias, 0)

    def forward(self, x, t, y):
        inputs = torch.cat([x, t, y], dim=1)
        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            for layer in self.layers:
                x = layer(torch.relu(h1))
        else:
            h1 = torch.relu(self.fc1(inputs))
            for layer in self.layers:
                x = layer(torch.relu(h1))

        h31 = torch.relu(self.fc31(x))
        logit1 = self.fc32(h31)
        y_hat_1 = torch.nn.Sigmoid()(logit1)
        h41 = torch.relu(self.fc41(x))
        logit2 = self.fc42(h41)
        y_hat_2 = torch.nn.Sigmoid()(logit2)
        return torch.cat([y_hat_1, y_hat_2], dim=1)


class Discriminator(nn.Module):
    """Discriminator function.

    Args:
      - x: features
      - t: treatments
      - y: observed labels
      - hat_y: estimated counterfactuals

    Returns:
      - D_logit: estimated potential outcomes
    """
    def __init__(self, input_dim, h_dim, flag_dropout):
        super(Discriminator, self).__init__()
        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim + 2, h_dim) # +2 for t and y
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.dp2_1 = nn.Dropout(p=0.2)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.dp2_2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(h_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.xavier_uniform_(self.fc2_1.weight)
        nn.init.constant_(self.fc2_1.bias, 0)
        nn.init.xavier_uniform_(self.fc2_2.weight)
        nn.init.constant_(self.fc2_2.bias, 0)

    def forward(self, x, t, y, hat_y):
        input0 = (1. - t) * y + t * hat_y[:, 0].unsqueeze(1) # if t = 0 dim=btx1
        input1 = t * y + (1. - t) * hat_y[:, 1].unsqueeze(1) # if t = 1
        inputs = torch.cat([x, input0, input1], dim=1)

        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            h2_1 = self.dp2_1(torch.relu(self.fc2_1(h1)))
            h2_2 = self.dp2_2(torch.relu(self.fc2_2(h2_1)))
            h2 = self.dp2(torch.relu(self.fc2(h2_2)))
        else:
            h1 = torch.relu(self.fc1(inputs))
            h2_1 = torch.relu(self.fc2_1(h1))
            h2_2 = torch.relu(self.fc2_2(h2_1))
            h2 = torch.relu(self.fc2(h2_2))

        return self.fc3(h2)

class InferenceNet(nn.Module):
    """Inference function.
    Args:
      - x: features
    Returns:
      - I_logit: estimated potential outcomes
    """
    def __init__(self, input_dim, h_dim, flag_dropout):
        super(InferenceNet, self).__init__()
        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim, h_dim)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.dp2 = nn.Dropout(p=0.2)

        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.dp2_1 = nn.Dropout(p=0.2)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.dp2_2 = nn.Dropout(p=0.2)

        # Output: Estimated outcome when t = 0
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)

        # Output: Estimated outcome when t = 1
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc31.weight)
        nn.init.constant_(self.fc31.bias, 0)
        nn.init.xavier_uniform_(self.fc32.weight)
        nn.init.constant_(self.fc32.bias, 0)
        nn.init.xavier_uniform_(self.fc41.weight)
        nn.init.constant_(self.fc41.bias, 0)
        nn.init.xavier_uniform_(self.fc42.weight)
        nn.init.constant_(self.fc42.bias, 0)
        nn.init.xavier_uniform_(self.fc2_1.weight)
        nn.init.constant_(self.fc2_1.bias, 0)
        nn.init.xavier_uniform_(self.fc2_2.weight)
        nn.init.constant_(self.fc2_2.bias, 0)

    def forward(self, x):
        inputs = x
        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            h2_1 = self.dp2_1(torch.relu(self.fc2_1(h1)))
            h2_2 = self.dp2_2(torch.relu(self.fc2_2(h2_1)))
            h2 = self.dp2(torch.relu(self.fc2(h2_2)))
        else:
            h1 = torch.relu(self.fc1(inputs))
            h2_1 = torch.relu(self.fc2_1(h1))
            h2_2 = torch.relu(self.fc2_2(h2_1))
            h2 = torch.relu(self.fc2(h2_2))

        # t = 0
        h31 = torch.relu(self.fc31(h2))
        logit1 = self.fc32(h31)
        y_hat_1 = torch.nn.Sigmoid()(logit1)

        # t = 1
        h41 = torch.relu(self.fc41(h2))
        logit2 = self.fc42(h41)
        y_hat_2 = torch.nn.Sigmoid()(logit2)
        return torch.cat([y_hat_1, y_hat_2], dim=1)

class InferenceNetDeep(nn.Module):
    """Inference function.
    Args:
      - x: features
    Returns:
      - I_logit: estimated potential outcomes
    """
    def __init__(self, input_dim, h_dim, flag_dropout, k):
        super(InferenceNetDeep, self).__init__()
        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim, h_dim)
        self.dp1 = nn.Dropout(p=0.2)

        self.layers = nn.ModuleList()
        for _ in range(k-3):
            layer = nn.Linear(h_dim, h_dim)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
            self.layers.append(layer)
            dp = nn.Dropout(p=0.2)
            self.layers.append(dp)

        # Output: Estimated outcome when t = 0
        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)

        # Output: Estimated outcome when t = 1
        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc31.weight)
        nn.init.constant_(self.fc31.bias, 0)
        nn.init.xavier_uniform_(self.fc32.weight)
        nn.init.constant_(self.fc32.bias, 0)
        nn.init.xavier_uniform_(self.fc41.weight)
        nn.init.constant_(self.fc41.bias, 0)
        nn.init.xavier_uniform_(self.fc42.weight)
        nn.init.constant_(self.fc42.bias, 0)

    def forward(self, x):
        inputs = x
        if self.flag_dropout:
            h1 = self.dp1(torch.relu(self.fc1(inputs)))
            for layer in self.layers:
                x = layer(torch.relu(h1))
        else:
            h1 = torch.relu(self.fc1(inputs))
            for layer in self.layers:
                x = layer(torch.relu(h1))

        h31 = torch.relu(self.fc31(x))
        logit1 = self.fc32(h31)
        y_hat_1 = torch.nn.Sigmoid()(logit1)
        h41 = torch.relu(self.fc41(x))
        logit2 = self.fc42(h41)
        y_hat_2 = torch.nn.Sigmoid()(logit2)
        return torch.cat([y_hat_1, y_hat_2], dim=1)
