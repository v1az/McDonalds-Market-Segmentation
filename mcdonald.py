import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

mcdonalds = pd.read_csv('mcdonalds.csv')


print(mcdonalds.columns)


print(mcdonalds.shape)


print(mcdonalds.head(3))


mcdonalds.replace('Yes', 1, inplace=True)
mcdonalds.replace('No', 0, inplace=True)


print(round(mcdonalds.mean(), 2))


pca = PCA()
MD_pca = pca.fit_transform(mcdonalds.iloc[:, 0:11])


print(pca.explained_variance_ratio_)


plt.scatter(MD_pca[:, 0], MD_pca[:, 1], c='grey')
plt.show()


kmeans = KMeans(n_clusters=4, random_state=1234)
kmeans.fit(mcdonalds)
clusters_kmeans = kmeans.labels_


gmm = GaussianMixture(n_components=4, random_state=1234)
clusters_gmm = gmm.fit_predict(mcdonalds)


print(pd.crosstab(clusters_kmeans, clusters_gmm))


tree = DecisionTreeClassifier()
tree.fit(mcdonalds[['Like', 'Age', 'VisitFrequency', 'Gender']], clusters_kmeans)


plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True)
plt.show()


visit_mean = mcdonalds.groupby(clusters_kmeans)['VisitFrequency'].mean()
like_mean = mcdonalds.groupby(clusters_kmeans)['Like'].mean()
female_mean = mcdonalds.groupby(clusters_kmeans)['Gender'].apply(lambda x: (x == 'Female').mean())


plt.figure(figsize=(10, 6))
plt.scatter(visit_mean, like_mean, s=10 * female_mean, alpha=0.5)
for i in range(4):
    plt.text(visit_mean[i], like_mean[i], str(i+1))
plt.xlim(2, 4.5)
plt.ylim(-3, 3)
plt.xlabel('Visit Frequency')
plt.ylabel('Like')
plt.show()
