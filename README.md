# CryptoClustering

# Import required libraries and dependencies
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data into a Pandas DataFrame and make the index the "coin_id" column.
market_data_df = pd.read_csv("Resources/crypto_market_data.csv", index_col="coin_id")

# Display sample data
market_data_df.head(10)

# Generate summary statistics
market_data_df.describe()

# Use the `StandardScaler()` module from scikit-learn to normalize the data from the CSV file
market_data_scaled = StandardScaler().fit_transform(market_data_df)  

market_data_scaled[:5]

#Create a DataFrame with the scaled data
#transformed_market_data_df = pd.DataFrame(
#    market_data_scaled)
scaled_market_data_df = pd.DataFrame(
    market_data_scaled, columns=market_data_df.columns, index=market_data_df.index
 )

# Copy the crypto names from the original data
scaled_market_data_df['coin_id'] = market_data_df.index

# Set the coinid column as index
transformed_market_data_df = scaled_market_data_df.set_index('coin_id')


# Display sample data
transformed_market_data_df.head(10)

# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))  

# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using the scaled DataFrame
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(transformed_market_data_df)
    inertia.append(model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data = {"k": k, 
              "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve
elbow_data_df = pd.DataFrame(elbow_data)

# Display the DataFrame
elbow_data_df

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_data_df.plot.line(x='k', y='inertia', title='Elbow Curve', xticks=k)

#### Answer the following question: 
**Question:** What is the best value for `k`?

**Answer:** It looks like the best value for K is 4.

# Initialize the K-Means model using the best value for k
kmeans_model_1 = KMeans(n_clusters=4, random_state=0)

# Fit the K-Means model using the scaled data
kmeans_model_1.fit(transformed_market_data_df)

# Predict the clusters to group the cryptocurrencies using the scaled data
predictions = kmeans_model_1.predict(transformed_market_data_df)

# View the resulting array of cluster values
print(predictions)

# Create a copy of the DataFrame
market_predictions_df = transformed_market_data_df.copy()

# Add a new column to the DataFrame with the predicted clusters
market_predictions_df['market predictions'] = predictions

# Display sample data
market_predictions_df.head(5)

# Create a scatter plot using Pandas plot by setting 
# `x="price_change_percentage_24h"` and `y="price_change_percentage_7d"`.
# Use "rainbow" for the color to better visualize the data.
market_predictions_df.plot.scatter(x='price_change_percentage_24h', y='price_change_percentage_7d', c='market predictions', colormap='rainbow', title='Clusters of Cryptocurrencies')

# Create a PCA model instance and set `n_components=3`.
market_pca = PCA(n_components=3)

# Use the PCA model with `fit_transform` on the original scaled DataFrame to reduce to three principal components.
market_pca_data = market_pca.fit_transform(transformed_market_data_df)  

# View the first five rows of the DataFrame. 
market_pca_data[:5]

# Retrieve the explained variance to determine how much information  can be attributed to each principal component.
market_pca_data_variance = market_pca.explained_variance_ratio_
market_pca_data_variance

#### Answer the following question: 

**Question:** What is the total explained variance of the three principal components?

**Answer:** PCA1 is about 37%, decreasing to about 34% and then decreasing again to about 17%.

# Create a new DataFrame with the PCA data.
# Note: The code for this step is provided for you
new_market_pca_data_df = pd.DataFrame(
    market_pca_data, columns=["PCA1", "PCA2", "PCA3"]
)
# Creating a DataFrame with the PCA data


# Copy the crypto names from the original data
new_market_pca_data_df['coin_id'] = market_data_df.index

# Set the coinid column as index
new_market_pca_data_df = new_market_pca_data_df.set_index('coin_id')

# Display sample data
new_market_pca_data_df.head(5)

# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))  

# Create an empty list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using PCA DataFrame.
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(new_market_pca_data_df)
    inertia.append(model.inertia_)

# Create a dictionary with the data to plot the Elbow curve
elbow_data_2 = {"k": k, 
                "inertia": inertia}

# Create a DataFrame with the data to plot the Elbow curve
elbow_data_2_df = pd.DataFrame(elbow_data_2)

# Display the DataFrame
elbow_data_2_df

# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_data_2_df.plot.line(x='k', y='inertia', title='Elbow Curve', xticks=k)  

#### Answer the following questions: 
* **Question:** What is the best value for `k` when using the PCA data?

  * **Answer:** The best value for 'k' appears to be 4 when using the PCA data.


* **Question:** Does it differ from the best k value found using the original data?

  * **Answer:** The best value for 'k' using PCA looks the same as the best value for 'k' using the original data.

# Initialize the K-Means model using the best value for k
pca_model_1 = KMeans(n_clusters=4, random_state=0)

# Fit the K-Means model using the PCA data
pca_model_1.fit(transformed_market_data_df)

# Predict the clusters to group the cryptocurrencies using the PCA data
pca_predictions = kmeans_model_1.predict(transformed_market_data_df)

# Print the resulting array of cluster values.
pca_predictions

# Create a copy of the DataFrame with the PCA data
pca_market_predictions_df = pca_predictions.copy()

# Add a new column to the DataFrame with the predicted clusters
new_market_pca_data_df['pca predictions'] = pca_predictions
#pca_market_predictions_df = pd.DataFrame(pca_market_predictions_df, columns=["market predictions"])

# Display sample data
new_market_pca_data_df.head()

# Create a scatter plot using hvPlot by setting `x="PCA1"` and `y="PCA2"`. 
new_market_pca_data_df.plot.scatter(x='PCA1', y='PCA2', c='pca predictions', colormap='winter', title='crypto clusters')

# Use the columns from the original scaled DataFrame as the index.
pca_component_weights_df = pd.DataFrame(market_pca.components_.T, columns=["PCA1", "PCA2", "PCA3"], index=transformed_market_data_df.columns)
pca_component_weights_df

#### Answer the following question: 

* **Question:** Which features have the strongest positive or negative influence on each component? 
 
* **Answer:** price_change_percentage_200d has the strongest positive influence on PCA1, price_change_percentage_24h has the strongest negative influence on PCA1.  On PCA2, price_change_percentage_30d has the strongest positive influence and price_change_percentage_1y has the strongest negative influence. Lastly, price_change_percentage_7d has the strongest positive influence and price_change_percentage_60d has the strongest negative influence.  The strongest influence overall is price_change_percentage_7d in the PCA3 column and the strongest negative inflence overall is price_change_percentage_24h in the PCA1 column.

