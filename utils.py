import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from IPython.display import display_html
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA


def render_subdfs(df, chunk_size=4):
    """Example Usage:
        stats_df = pd.DataFrame({
            'Skewness': df.skew(numeric_only=True),
            'Kurtosis': df.kurtosis(numeric_only=True),
            'Null Count': df.isnull().sum()
        })
        render_subdfs(stats_df, chunk_size=9)
    """
    chunks = [
        df.iloc[i:i + chunk_size] # Divide a DataFrame into smaller chunks, 
        for i in range(0, df.shape[0], chunk_size) # Each of which can be processed/displayed separately
    ]
    flex_container = f'''
        <div style="display: flex; justify-content: left; flex-wrap: wrap;">
            {''.join([ 
                chunk.to_html().replace('table', 'table style="display:inline-block; padding: 5px; margin: 5px"') 
                for chunk in chunks
            ])}
        </div>
    ''' # Convert each chunk to HTML and then combine them for side-by-side display
    display_html(flex_container, raw=True)
    
    
def detect_outliers(feature):
    if not np.issubdtype(feature.dtype, np.number): return None, None, None
    Q1, Q3 = np.nanpercentile(feature, [25, 75]) # Calculate the 1st and 3rd quartiles without considering NaN values
    IQR = Q3 - Q1
    
    # Define outliers as those beyond 1.5 * IQR from the Q1 and Q3
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    is_outliers = (feature < lower_bound) | (feature > upper_bound)
    return is_outliers, lower_bound, upper_bound


def stats_summary(df):
    summary = df.describe(include=[np.number]).T # Calculate descriptive statistics for all numerical attributes
    summary['variance'] = df.var(numeric_only=True) # Variance = E[(X - E[X])^2]
    summary['iqr_size'] = df.quantile(0.75, numeric_only=True) - df.quantile(0.25, numeric_only=True) # IQR = Q3 - Q1
    summary['skewness'] = df.skew(numeric_only=True) # Skew > 0 => Right-skewed distribution
    summary['kurtosis'] = df.kurtosis(numeric_only=True) # Kurtosis > 0 => This distribution is sharper than the normal distribution
    summary['nulls_count'] = df.isnull().sum() # Checking for null values to identify attributes with missing values
    summary['outliers_count'] = df.select_dtypes(include=[np.number]).apply(lambda col: detect_outliers(col)[0].sum())
    summary['nulls_percent'] = summary['nulls_count'] * 100 / df.shape[0]
    summary['outliers_percent'] = summary['outliers_count'] * 100 / summary['count']
    return summary


def hist_box_plot(figsize, X, list_of_cols, ncols=1, xlabel='Data Values', ylabel='Frequency', plt_show=True):
    fig = plt.figure(figsize=figsize)
    outer_grid = gridspec.GridSpec(len(list_of_cols) // ncols + 1, ncols)
    
    for i, col_name in enumerate(list_of_cols):
        inner_grid = outer_grid[i].subgridspec(2, 1, height_ratios=[5, 1], hspace=0)
        ax_hist = fig.add_subplot(inner_grid[0])
        ax_box = fig.add_subplot(inner_grid[1])
        
        feature = X[col_name]
        feature_median, feature_min, feature_max = feature.median(), feature.min(), feature.max()

        # Plot the distribution for frequency analysis
        sns.histplot(feature, ax=ax_hist, kde=True, color="skyblue", edgecolor='red')
        ax_hist.set_xlabel('')  # Remove the x-axis label
        ax_hist.set_xticks([]) # Remove x-ticks for histogram
        ax_hist.set_ylabel(ylabel)
        ax_hist.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)
        ax_hist.set_title(f'Distribution with Histogram and Boxplot for {col_name}', fontweight='bold')
        
        # Draw a vertical line at the median
        ax_hist.axvline(feature_median, ymax=0.9, linestyle='--', color='green')
        ax_hist.annotate(
            f'Median: {feature_median:.2f}',
            xy=(feature_median, 0), xytext=(0.3, 0.5), textcoords='axes fraction',
            arrowprops={'arrowstyle': '->', 'color': 'green', 'linewidth': 1.5}
        )
        
        # Plot the boxplot for outlier detection
        sns.boxplot(feature, ax=ax_box, width=0.5, color='lightgreen', fliersize=5, linewidth=1.5, orient='h')
        _, lower_bound, upper_bound = detect_outliers(feature)
        
        # Show the range considered as outliers
        ax_box.axvspan(xmin=feature_min, xmax=feature_max, color='green', alpha=0.2)
        if lower_bound > feature_min: ax_box.axvspan(xmin=feature_min, xmax=lower_bound, color='red', alpha=0.3)
        if upper_bound < feature_max: ax_box.axvspan(xmin=upper_bound, xmax=feature_max, color='red', alpha=0.3)
        
        ax_box.set_xlabel(xlabel)
        ax_box.set_ylabel('')
        ax_box.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)

    if plt_show: 
        plt.tight_layout()
        plt.show()
    else: return fig, outer_grid
    
    
def winsorize_outliers(X, limits=[0.05, 0.05]):
    X_winsorized = X.copy()
    before_winsorize = stats_summary(X_winsorized)[['outliers_count', 'outliers_percent']]
    before_winsorize = before_winsorize[before_winsorize.outliers_count > 0]
    
    cols_with_outliers = before_winsorize.index
    X_winsorized[cols_with_outliers] = X_winsorized[cols_with_outliers].apply(lambda col: winsorize(col, limits=limits)) 
    after_winsorize = stats_summary(X_winsorized.loc[:, cols_with_outliers])[['outliers_count', 'outliers_percent']]
    
    # Concatenate the 2 DataFrames for comparison
    outliers_summary = pd.concat([
        before_winsorize.rename(columns={'outliers_count': 'outliers_count_before', 'outliers_percent': 'outliers_percent_before'}), 
        after_winsorize.rename(columns={'outliers_count': 'outliers_count_after', 'outliers_percent': 'outliers_percent_after'})
    ], axis=1).sort_values('outliers_percent_after', ascending=False)
    return X_winsorized, outliers_summary


def pca_grouping_train(X_train, pca_features={}): # PCA for .fit_transform() on training data
    # Drop original features and concatenate the PCA components
    X_train_new = [X_train.drop(columns=sum(pca_features.values(), []))] # Drop original features
    pca_dict = {} # Store the PCA objects for .transform() on testing data

    # Applying PCA to reduce dimensionality while trying to retain most of the variance in the data
    for group_name, feature_names in pca_features.items():
        pca = PCA(n_components=1) # Reduce to 1 component for simplicity
        pca_feature_train = pca.fit_transform(X_train[feature_names])
        pca_feature_train = pd.DataFrame(pca_feature_train, columns=[group_name], index=X_train.index)

        pca_dict[group_name] = pca # Store the PCA object for .transform() on testing data
        X_train_new.append(pca_feature_train) 
        print(f'The new {group_name} feature explains {pca.explained_variance_ratio_[0]*100:.2f}% of the variance')
    return pca_dict, pd.concat(X_train_new, axis=1)


def pca_grouping_test(X_test, pca_dict, pca_features={}): # PCA for .transform() on testing data
    X_test_new = [X_test.drop(columns=sum(pca_features.values(), []))]
    for group_name, feature_names in pca_features.items():
        pca_feature_test = pca_dict[group_name].transform(X_test[feature_names])
        pca_feature_test = pd.DataFrame(pca_feature_test, columns=[group_name], index=X_test.index)
        X_test_new.append(pca_feature_test)
    return pd.concat(X_test_new, axis=1)