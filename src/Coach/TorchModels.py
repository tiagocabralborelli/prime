import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


class _MCMLP(nn.Module):
    """
    Internal PyTorch neural network architecture with Dropout layers
    between each hidden representation.
    """
    def __init__(self, input_dim, hidden_layer_sizes, num_classes, dropout_rate=0.2, activation='relu'):
        super().__init__()
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())

            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """
    A PyTorch-based Multi-Layer Perceptron (MLP) Classifier compatible with scikit-learn
    pipelines (GridSearchCV, clone, cross_val_score) that supports Monte Carlo Dropout
    for predictive uncertainty estimation.
    """
    def __init__(
        self,
        hidden_layer_sizes=(64,),
        dropout_rate=0.2,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=32,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        activation='relu',
        random_state=None,
        device='cpu',
        mc_samples=50,
        use_mc_dropout=False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.activation = activation
        self.random_state = random_state
        self.device = device
        self.mc_samples = mc_samples
        self.use_mc_dropout = use_mc_dropout

    def _get_device(self):
        if self.device == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X, y):
        self._set_seed()
        device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        # Map classes to integers 0, 1, ..., C-1
        label_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_indices = np.array([label_to_idx[val] for val in y], dtype=np.int64)

        if self.early_stopping and len(X) > 10 and self.validation_fraction > 0:
            stratify = y_indices if np.min(np.bincount(y_indices)) >= 2 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_indices,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify
            )
        else:
            X_train, y_train = X, y_indices
            X_val, y_val = None, None

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        batch_size = min(self.batch_size, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model_ = _MCMLP(
            input_dim=self.n_features_in_,
            hidden_layer_sizes=self.hidden_layer_sizes,
            num_classes=self.n_classes_,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        self.loss_curve_ = []
        self.val_loss_curve_ = []

        for epoch in range(self.max_iter):
            self.model_.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(batch_x)

            epoch_train_loss = running_loss / len(train_dataset)
            self.loss_curve_.append(epoch_train_loss)

            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                    val_y_t = torch.tensor(y_val, dtype=torch.long).to(device)
                    val_outputs = self.model_(val_x_t)
                    val_loss = criterion(val_outputs, val_y_t).item()

                self.val_loss_curve_.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model_.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.n_iter_no_change:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.is_fitted_ = True
        return self

    def _enable_mc_dropout(self):
        """Sets the model in eval mode while keeping all Dropout layers in train mode."""
        self.model_.eval()
        for module in self.model_.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_mc_dropout(self, X, mc_samples=None):
        """
        Runs Monte Carlo Dropout passes to estimate prediction probabilities and uncertainty.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        mc_samples : int, optional (defaults to self.mc_samples)

        Returns
        -------
        mean_proba : ndarray of shape (n_samples, n_classes)
            Average predictive probabilities across MC samples.
        std_proba : ndarray of shape (n_samples, n_classes)
            Standard deviation across MC samples (epistemic uncertainty).
        all_samples : ndarray of shape (mc_samples, n_samples, n_classes)
            Raw predicted probabilities for each MC pass.
        """
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError("TorchMLPClassifier is not fitted yet.")

        n_samples = mc_samples or self.mc_samples
        device = self._get_device()
        self._enable_mc_dropout()

        X = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        mc_probas = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model_(X_t)
                probas = torch.softmax(logits, dim=-1).cpu().numpy()
                mc_probas.append(probas)

        mc_probas = np.array(mc_probas) # shape: (mc_samples, n_samples, n_classes)
        mean_proba = np.mean(mc_probas, axis=0)
        std_proba = np.std(mc_probas, axis=0, ddof=1) if n_samples > 1 else np.zeros_like(mean_proba)

        return mean_proba, std_proba, mc_probas

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        If use_mc_dropout is True, uses the mean probability of MC Dropout passes.
        Otherwise, uses standard deterministic evaluation.
        """
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError("TorchMLPClassifier is not fitted yet.")

        if self.use_mc_dropout:
            mean_proba, _, _ = self.predict_mc_dropout(X, mc_samples=self.mc_samples)
            return mean_proba

        device = self._get_device()
        self.model_.eval()
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self.model_(X_t)
            probas = torch.softmax(logits, dim=-1).cpu().numpy()

        return probas

    def predict(self, X):
        """Predict class labels for X."""
        probas = self.predict_proba(X)
        pred_idx = np.argmax(probas, axis=-1)
        return self.classes_[pred_idx]


class _LinearClassifierNet(nn.Module):
    """Single linear layer for Logistic Regression in PyTorch."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class TorchLogisticRegression(ClassifierMixin, BaseEstimator):
    """
    A PyTorch-based Logistic Regression Classifier compatible with scikit-learn
    pipelines (GridSearchCV, clone, cross_val_score).

    Because it uses stochastic mini-batch optimization and seeded weight
    initialization, it is sensitive to random_state / seed, making it ideal
    for measuring uncertainty across seed runs.
    """
    def __init__(
        self,
        C=1.0,
        learning_rate=0.01,
        batch_size=32,
        max_iter=100,
        class_weight='balanced',
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=None,
        device='cpu',
    ):
        self.C = C
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.device = device

    def _get_device(self):
        if self.device == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def fit(self, X, y):
        self._set_seed()
        device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        label_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_indices = np.array([label_to_idx[val] for val in y], dtype=np.int64)

        if self.early_stopping and len(X) > 10 and self.validation_fraction > 0:
            stratify = y_indices if np.min(np.bincount(y_indices)) >= 2 else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_indices,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify
            )
        else:
            X_train, y_train = X, y_indices
            X_val, y_val = None, None

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        batch_size = min(self.batch_size, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model_ = _LinearClassifierNet(
            input_dim=self.n_features_in_,
            num_classes=self.n_classes_
        ).to(device)

        # Class weights calculation
        if self.class_weight == 'balanced':
            counts = np.bincount(y_train, minlength=self.n_classes_)
            total = len(y_train)
            weights = total / (self.n_classes_ * np.maximum(counts, 1).astype(float))
            weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        # Regularization: C is inverse of regularization strength
        weight_decay = 1.0 / max(float(self.C), 1e-6)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay
        )

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        self.loss_curve_ = []
        self.val_loss_curve_ = []

        for epoch in range(self.max_iter):
            self.model_.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(batch_x)

            epoch_train_loss = running_loss / len(train_dataset)
            self.loss_curve_.append(epoch_train_loss)

            if X_val is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_x_t = torch.tensor(X_val, dtype=torch.float32).to(device)
                    val_y_t = torch.tensor(y_val, dtype=torch.long).to(device)
                    val_outputs = self.model_(val_x_t)
                    val_loss = criterion(val_outputs, val_y_t).item()

                self.val_loss_curve_.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model_.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.n_iter_no_change:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Expose linear weights and intercept similar to sklearn LogisticRegression
        with torch.no_grad():
            self.coef_ = self.model_.linear.weight.detach().cpu().numpy()
            self.intercept_ = self.model_.linear.bias.detach().cpu().numpy()

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        if not getattr(self, 'is_fitted_', False):
            raise RuntimeError("TorchLogisticRegression is not fitted yet.")

        device = self._get_device()
        self.model_.eval()
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = self.model_(X_t)
            probas = torch.softmax(logits, dim=-1).cpu().numpy()

        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        pred_idx = np.argmax(probas, axis=-1)
        return self.classes_[pred_idx]

