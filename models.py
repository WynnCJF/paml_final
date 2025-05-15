import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler

# --- Logistic Regression Class ---
class LogisticRegression(object):
    def __init__(self, learning_rate=0.001, num_iterations=500):
        self.model_name = 'Logistic Regression'
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.likelihood_history=[]
        self.loss_history = []

    def predict_probability(self, X):
        """
        Produces probabilistic estimate for P(y_i = +1 | x_i, w)
        """
        y_pred=None
        try:
            score = X.dot(self.W) + self.b
            y_pred = 1. / (1. + np.exp(-score))
        except ValueError as err:
            print({str(err)})
        return y_pred

    def compute_avg_log_likelihood(self, X, Y, W):
        """
        Compute the average log-likelihood of logistic regression coefficients
        """
        lp=None
        try:
            indicator = (Y == +1)
            scores = np.dot(X, W)
            logexp = np.log(1. + np.exp(-scores))

            mask = np.isinf(logexp)
            logexp[mask] = -scores[mask]
            lp = np.sum((indicator - 1) * scores - logexp) / len(X)
        except ValueError as err:
            print({str(err)})
        return lp
    
    def compute_loss(self, X, Y):
        """
        Compute the binary cross-entropy loss
        """
        try:
            m = X.shape[0]
            y_pred = self.predict_probability(X)
            y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
            loss = (-1/m) * np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))
            return loss
        except ValueError as err:
            print({str(err)})
            return 0

    def update_weights(self):
        """
        Compute the logistic regression derivative and update weights
        """
        try:
            num_examples = self.X.shape[0]
            y_pred_prob = self.predict_probability(self.X)
            
            dW = self.X.T.dot(self.Y - y_pred_prob) / num_examples
            db = np.sum(self.Y - y_pred_prob) / num_examples

            self.b = self.b + self.learning_rate * db
            self.W = self.W + self.learning_rate * dW

            lp = self.compute_avg_log_likelihood(self.X, self.Y, self.W)
            self.likelihood_history.append(lp)
            
            loss = self.compute_loss(self.X, self.Y)
            self.loss_history.append(loss)
        except ValueError as err:
            print({str(err)})
        return self

    def predict(self, X):
        """
        Predicts class labels based on logistic regression decision boundary
        """
        y_pred=None
        try:
            Z = self.predict_probability(X)
            y_pred = [0 if z <= 0.5 else +1 for z in Z]
        except ValueError as err:
                print({str(err)})
        return y_pred

    def fit(self, X, Y, callback=None):
        """
        Fits the logistic regression model to the data using gradient ascent
        """
        self.W = np.random.randn(X.shape[1]) * 0.01
        self.X = X
        self.Y = Y
        self.b = 0
        self.likelihood_history=[]
        self.loss_history = []
        
        try:
            for i in range(self.num_iterations):
                self.update_weights()
                
                if callback and i % 100 == 0:
                    progress = int((i + 1) / self.num_iterations * 100)
                    callback(i, progress, self.loss_history[-1])
                    
        except ValueError as err:
                print({str(err)})
        return self

# --- Custom SVM Class ---
class CustomSVM:
    def __init__(self,
                 learning_rate=0.001,
                 num_iterations=1000,
                 lambda_param=0.01,
                 weighted_loss=True
                 ):
        self.model_name = 'Custom SVM Implementation'
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.likelihood_history = []
        self.loss_history = []
        self.weighted_loss = weighted_loss

    def predict_score(self, X):
        scores = np.dot(X, self.W) + self.b
        return scores

    def compute_hinge_loss(self, X, Y):
        Y_adjusted = np.where(Y <= 0, -1, 1)
        margins = 1 - Y_adjusted * self.predict_score(X)
        margins = np.maximum(0, margins)
        loss = np.mean(margins) + (self.lambda_param / 2) * np.sum(self.W ** 2)
        return loss

    def update_weights(self):
        num_examples = self.X.shape[0]
        Y_adjusted = np.where(self.Y <= 0, -1, 1)
        scores = self.predict_score(self.X)

        if self.weighted_loss:
            pos_weight = self.Y.mean()
            neg_weight = 1 - pos_weight

            sample_weights = np.where(self.Y > 0, 1/pos_weight, 1/neg_weight)
            indicator = (Y_adjusted * scores) < 1

            dW = (- (self.X.T * sample_weights).dot(Y_adjusted * indicator) + self.lambda_param * self.W) / num_examples
            db = -np.sum(sample_weights * Y_adjusted * indicator) / num_examples
        else:
            indicator = (Y_adjusted * scores) < 1
            dW = (-np.dot(self.X.T, (Y_adjusted * indicator)) + self.lambda_param * self.W) / num_examples
            db = -np.sum(Y_adjusted * indicator) / num_examples

        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

        loss = self.compute_hinge_loss(self.X, self.Y)
        self.loss_history.append(loss)
        self.likelihood_history.append(-loss)

        return self

    def predict(self, X, threshold=0.0):
        scores = self.predict_score(X)
        y_pred = np.where(scores < threshold, 0, 1)
        return y_pred

    def fit(self, X, Y, callback=None):
        self.num_examples, self.num_features = X.shape
        self.X = X
        self.Y = Y
        self.W = np.random.randn(self.num_features) * 0.01
        self.b = 0
        self.likelihood_history = []
        self.loss_history = []

        for i in range(self.num_iterations):
            self.update_weights()
            
            if callback and i % 100 == 0:
                progress = int((i + 1) / self.num_iterations * 100)
                callback(i, progress, self.loss_history[-1])

        return self

# --- Data Preprocessing Functions ---
def normalize_data(df):
    """
    Apply normalization to the diabetes dataset features
    """
    numeric_features_for_robust_normalization = [
        "BMI",
        "MentHlth",
        "PhysHlth",
    ]

    numeric_features_for_standard_normalization = [
        "Age",
        "Income",
        "Education",
        "GenHlth",
    ]

    df_normalized = df.copy()

    robust_scaler = RobustScaler()
    df_normalized[numeric_features_for_robust_normalization] = robust_scaler.fit_transform(df_normalized[numeric_features_for_robust_normalization])

    std_scaler = StandardScaler()
    df_normalized[numeric_features_for_standard_normalization] = std_scaler.fit_transform(df_normalized[numeric_features_for_standard_normalization])
    
    return df_normalized

def prepare_data(df, test_size=0.3, feature_cols=None):
    """
    Prepare the diabetes dataset, normalize features and split into train/test sets
    
    Parameters:
    - df: Pandas DataFrame containing the dataset
    - test_size: Proportion of the dataset to include in the test split
    - feature_cols: Optional list of feature column names to use (for feature selection)
    
    Returns:
    - X_train, X_test, y_train, y_test: Train/test splits
    """
    df_normalized = normalize_data(df)
    y_data = df_normalized['Diabetes_binary']
    
    if feature_cols is not None:
        x_data = df_normalized[feature_cols]
    else:
        x_data = df_normalized.drop('Diabetes_binary', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def apply_resampling(X_train, y_train, method):
    """
    Apply the specified resampling method to the training data
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - method: Resampling method ("None", "SMOTE", "SMOTE + Undersampling", "ENN")
    
    Returns:
    - Resampled X_train, y_train, and a description of what was done
    """
    description = ""
    
    if method == "None":
        description = f"No resampling applied - Training set: {X_train.shape[0]} samples"
        return X_train, y_train, description
    
    elif method == "SMOTE":
        smote = SMOTE(random_state=42, sampling_strategy='auto', k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        description = f"Applied SMOTE resampling - Training set: {X_resampled.shape[0]} samples"
        
    elif method == "ENN":
        enn = EditedNearestNeighbours(kind_sel='all', n_neighbors=5)
        X_resampled, y_resampled = enn.fit_resample(X_train, y_train)
        description = f"Applied ENN resampling - Training set: {X_resampled.shape[0]} samples"
        
    elif method == "SMOTE + Undersampling":
        smote = SMOTE(sampling_strategy=0.6, k_neighbors=5, random_state=42)
        X_temp, y_temp = smote.fit_resample(X_train, y_train)
        
        rus = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_temp, y_temp)
        description = f"Applied SMOTE + Undersampling - Training set: {X_resampled.shape[0]} samples"
    
    else:
        description = f"No resampling applied (unknown method) - Training set: {X_train.shape[0]} samples"
        return X_train, y_train, description
    
    pos_count = sum(y_resampled == 1)
    neg_count = sum(y_resampled == 0)
    description += f" (Diabetes: {pos_count}, Non-Diabetes: {neg_count})"
    
    return X_resampled, y_resampled, description 