import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, PowerTransformer, PolynomialFeatures
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

mcc_scorer = make_scorer(matthews_corrcoef)

def optimize_threshold(y_true, y_probs):
    best_thresh = 0.5
    best_mcc = -1
    for thr in np.linspace(0,1,101):
        y_pred = (y_probs > thr).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        if score > best_mcc:
            best_mcc = score
            best_thresh = thr
    return best_thresh, best_mcc

class FeatureJittering(BaseEstimator, TransformerMixin):
    def __init__(self, noise_level=0.001):
        self.noise_level = noise_level
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X + np.random.normal(0, self.noise_level, size=X.shape)

class ClusterFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
    def fit(self, X, y=None):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X)
        return self
    def transform(self, X):
        clusters = self.kmeans.predict(X)
        return np.c_[X, clusters]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['target'] = (train['class'] == 'p').astype(int)
y = train['target'].values
X = train.drop(['id', 'class', 'target'], axis=1)

num_cols = ['cap-diameter', 'stem-height', 'stem-width']
cat_cols = [c for c in X.columns if c not in num_cols]

best_thresh = 0.5

encoder = ce.TargetEncoder(cols=cat_cols, smoothing=1)
X_enc = encoder.fit_transform(X, y)

numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()),
    ('jitter', FeatureJittering(noise_level=0.001))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor_init = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

X_prep = preprocessor_init.fit_transform(X_enc)
n_num = len(num_cols)
X_num = X_prep[:, :n_num]
X_cat = X_prep[:, n_num:]

clusterer = ClusterFeatures(n_clusters=5, random_state=42)
X_num_clustered = clusterer.fit_transform(X_num)

feature_engineering_numeric = Pipeline(steps=[
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('power', PowerTransformer(method='yeo-johnson'))
])
X_num_enh = feature_engineering_numeric.fit_transform(X_num_clustered)
X_full = np.hstack([X_num_enh, X_cat])

sel = VarianceThreshold(threshold=0.0)
X_full = sel.fit_transform(X_full)

print("training ensemble with 10 different boosting models is started")

models = []

# XGBoost модели
models.append(("xgb_1", xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, booster='gbtree',
                                          random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)))
models.append(("xgb_2", xgb.XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=4, booster='gbtree',
                                          random_state=43, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)))
models.append(("xgb_3", xgb.XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6, booster='gbtree',
                                          random_state=44, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)))
models.append(("xgb_rf", xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, subsample=0.8,
                                           colsample_bynode=0.8, random_state=45, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)))

# LightGBM модели
models.append(("lgb_1", lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=31, random_state=46, n_jobs=-1)))
models.append(("lgb_2", lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=64, random_state=47, n_jobs=-1)))


# CatBoost модели
models.append(("cat_1", cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, random_state=49, silent=True)))
models.append(("cat_2", cb.CatBoostClassifier(iterations=300, learning_rate=0.01, depth=8, random_state=50, silent=True)))
models.append(("cat_3", cb.CatBoostClassifier(iterations=300, learning_rate=0.03, depth=10, random_state=51, silent=True)))

# Gradient Boosting (sklearn)
models.append(("gb_1", GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, random_state=52)))
models.append(("gb_2", GradientBoostingClassifier(n_estimators=1000, learning_rate=0.03, max_depth=4, random_state=53)))


# ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier

models.append(("extrees_1", ExtraTreesClassifier(n_estimators=100, random_state=56, n_jobs=-1)))

# RandomForestClassifier
models.append(("rf_1", RandomForestClassifier(n_estimators=300, max_depth=5, random_state=58, n_jobs=-1)))

stacking_model = StackingClassifier(
    estimators=models,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    n_jobs=-1
)


stacking_model.fit(X_full, y)

print("it's all done")