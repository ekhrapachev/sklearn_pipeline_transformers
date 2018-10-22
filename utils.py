from sklearn.impute import SimpleImputer
from functools import reduce
from category_encoders import OrdinalEncoder, BinaryEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer

def check_df_values(self, df):
        result_df = df.dtypes.to_frame('dtype')
        result_df['num_unique'] = result_df.join(df.nunique().to_frame(name='num_unique'))['num_unique']
        
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'missing_values', 1 : '%_total_values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('%_total_values', ascending=False).round(1)

        result_df = result_df.merge(right=mis_val_table_ren_columns, left_index=True, right_index=True, how='left').fillna(0)
    
        return result_df.sort_values(by='dtype')

class ModelTransformer(TransformerMixin):
    def __init__(self, model, returnProba=False, featureName=None):
        self.model = model
        self.returnProba = returnProba
        self.feature_name = featureName
        if not self.feature_name: 
            self.feature_name = model.__class__.__name__

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def transform(self, X):
        if self.returnProba:
            x = pd.DataFrame(data={self.feature_name:[i[1] for i in self.model.predict_proba(X)]}, index=X.index)
        else:
            x = pd.DataFrame(data={self.feature_name:self.model.predict(X)}, index=X.index)
        return x

"""
DataFrame transformers. Using in sklearn pipeline
"""

class df_FeatureUnion(TransformerMixin):
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion
    
class df_ColumnSelector(TransformerMixin):
    def __init__(self, columns):
        if len(columns) == 1:
            self.columns = columns[0]
        else:
            self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
    
class df_Log1pTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xlog = np.log1p(X)
        return Xlog
    
class df_Imputer(TransformerMixin):
    """Imputation transformer for completing missing values
    
    Parameters
    ----------
    strategy: str (default='mean')
    -- 'mean'
    -- 'median'
    -- 'most_frequent'
    -- 'constant'
    
    fill_value: str of num (default=None)
    missing_values: number, string, np.nan (default) or None
    ----------
    """
    def __init__(self, strategy='mean', missing_values=np.NaN, fill_value=None):
        self.strategy = strategy
        self.missing_values = missing_values
        self.fill_value = fill_value
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):            
        self.imp = SimpleImputer(strategy=self.strategy, missing_values=self.missing_values, fill_value=self.fill_value)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled
    
class df_OneHotEncoder(TransformerMixin):
    def __init__(self, handle_unknown='ignore'):
        self.handle_unknown = handle_unknown
        
    def fit(self, X, y=None):
        self.enc = OneHotEncoder(handle_unknown=self.handle_unknown, sparse=False)
        self.enc.fit(X)
        self.feature_names = self.enc.get_feature_names(X.columns)
        return self
    
    def transform(self, X):
        X_encoded = self.enc.transform(X)
        X_encoded_df = pd.DataFrame(data=X_encoded, index=X.index, columns=self.feature_names)
        return X_encoded_df
    
class df_OrdinalEncoder(TransformerMixin):
    def __init__(self, handle_unknown='ignore'):
        self.handle_unknown = handle_unknown
        
    def fit(self, X, y=None):
        self.enc = OrdinalEncoder(handle_unknown=self.handle_unknown)
        self.enc.fit(X)
        return self
    
    def transform(self, X):
        X_encoded = self.enc.transform(X)
        X_encoded_df = pd.DataFrame(data=X_encoded, index=X.index, columns=X.columns)
        return X_encoded_df
    
class df_StandardScaler(TransformerMixin):
    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled
    
class df_Converter(TransformerMixin):
    def __init__(self, dtype, columns=None):
        self.dtype = dtype
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not self.columns:
            self.columns = list(X.columns)
            
        for i in self.columns:
            X[i] = X[i].astype(self.dtype)
        return X
    
class df_ExtractDateFeatures(TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not self.columns:
            self.columns = list(X.columns)

        for i in self.columns:
            X['{}_year'.format(i)] = X[i].dt.year
            X['{}_month'.format(i)] = X[i].dt.month
            X['{}_day'.format(i)] = X[i].dt.day
            X['{}_weekday'.format(i)] = X[i].dt.weekday
            X['{}_weekdayofyear'.format(i)] = X[i].dt.weekofyear
            X.drop(labels=i, inplace=True, axis=1)
        return X
    
class df_DateFormatter(TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.columns:
            self.columns = list(X.columns)
            
        Xdate = X[self.columns].apply(pd.to_datetime)
        return Xdate
    
class df_BinaryEncoder(TransformerMixin):
    """
    Use for encoding nominal features
    Parameters
    ----------
    handle_unknown: str, default='ignore'
    ----------
    """
    def __init__(self, handle_unknown='ignore'):
        self.handle_unknown = handle_unknown
        
    def fit(self, X, y=None):
        self.enc = BinaryEncoder(handle_unknown=self.handle_unknown)
        self.enc.fit(X)
        return self
    
    def transform(self, X):
        return self.enc.transform(X)
    
class df_DateDiffer():
    """
    Parameters:
    -----------
    pairs: list of tuples ([(col1,col2), (col3,col4)])
    -----------
    """
    def __init__(self, pairs=None):
        self.pairs = pairs
        
    def fit(self, X, y=None):
        return self
    
    def _timedelta(self, df):
        col1 = df.columns[:-1]
        col2 = df.columns[1:]
        Xbeg = df[col1].as_matrix()
        Xend = df[col2].as_matrix()
        Xd = (Xend - Xbeg) / np.timedelta64(1, 'h')
        diff_cols = ['->'.join(pair) for pair in zip(col1, col2)]
        Xdiff = pd.DataFrame(Xd, index=df.index, columns=diff_cols)
        return Xdiff

    def transform(self, X):
        if self.pairs:
            for i1, i2 in self.pairs:
                Xdiff = self._timedelta(X[[i1,i2]])
                X = pd.concat((X, Xdiff), axis=1)
        else:
            for i1, i2 in combinations(X.columns, 2):
                Xdiff = self._timedelta(X[[i1,i2]])
                X = pd.concat((X, Xdiff), axis=1)
                
        final_cols = [i for i in X.columns if '->' in i]
        return X[final_cols]
    
class df_DenseTransformer(TransformerMixin):        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()
    
class df_TfidfVectorizer(TransformerMixin):
    def __init__(self, max_features=2000, ngram_range=(1,1), stop_words=None, max_df=1.0, min_df=1):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words  = stop_words
        self.max_df = max_df
        self.min_df = min_df
        
        
    def fit(self, X, y=None):
        self.tfidf = TfidfVectorizer(max_features=self.max_features, ngram_range=self.ngram_range, stop_words=self.stop_words, max_df=self.max_df, min_df=self.min_df)
        self.tfidf.fit(X)
        return self
    
    def transform(self, X):
        X_tfidf = self.tfidf.transform(X)
        X_tfidf_df = pd.DataFrame(data=X_tfidf.toarray(), index=X.index, columns=self.tfidf.get_feature_names())
        return X_tfidf_df