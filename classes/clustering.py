import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift
from sklearn.mixture import GaussianMixture
from kmodes.kmodes import KModes
from sklearn.manifold import MDS
import math

SEED = 42

class Clustering:
    def print_calinski_harabasz_score(X_norm, clusters_predicted):
        metric_CH = metrics.calinski_harabasz_score(X_norm, clusters_predicted)
        print("Calinski-Harabasz Index: {:.3f}".format(metric_CH))
        return
    
    def print_silhouette_score(X, X_norm, clusters_predicted):
        # Get a sample to calculate silhouette coefficient
        muestra_silhoutte = 0.5 if (len(X) > 10000) else 1.0 
        metric_SC = metrics.silhouette_score(X_norm, clusters_predicted, metric='euclidean', sample_size=math.floor(muestra_silhoutte*len(X)), random_state=123456)
        print("Silhouette Coefficient: {:.5f}".format(metric_SC))

        metric_SC_samples = metrics.silhouette_samples(X_norm, clusters_predicted, metric='euclidean')

        return metric_SC_samples


    def plot_centroids_heatmap(X, centers, title):
        plt.figure()

        centers.index += 1

        centers_desnormal = centers.copy()
        for var in list(centers):
            centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())
        fig = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.2f')

        plt.xticks(rotation=30)
        fig.set_ylim(len(centers),0)
        fig.set_title(title)
        fig.figure.set_size_inches(6,5)
        centers.index -= 1

        return
    

    def plot_distribution_attribute_x_cluster(X_kmeans, attributes, num_attributes, num_clusters, centers, column_to_sort, plot_type, colors):
        plt.figure()
        mpl.style.use('default')
        fig, axes = plt.subplots(num_clusters, num_attributes, sharey=True, figsize=(5,5))
        fig.subplots_adjust(wspace=0, hspace=0)

        centers_sort = centers.sort_values(by=[column_to_sort])

        rango = []
        for j in range(num_attributes):
            rango.append([X_kmeans[attributes[j]].min(), X_kmeans[attributes[j]].max()])

        for i in range(num_clusters):
            c = centers_sort.index[i]
            dat_filt = X_kmeans.loc[X_kmeans['cluster']==c]
            for j in range(num_attributes):
                if plot_type == 'kdeplot':
                    ax = sns.kdeplot(x=dat_filt[attributes[j]], label="", fill=True, color=colors[c], ax=axes[i,j])
                elif plot_type == 'histplot':
                    ax = sns.histplot(x=dat_filt[attributes[j]], label="", color=colors[c], ax=axes[i,j], kde=True) # mejor si se usa weights de 'DB090'
                elif plot_type == 'boxplot':
                    ax = sns.boxplot(x=dat_filt[attributes[j]], notch=True, color=colors[c], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])

                ax.set(xlabel=attributes[j] if (i==num_clusters-1) else '', ylabel='Cluster '+str(c+1) if (j==0) else '')

                ax.set(yticklabels=[])
                ax.tick_params(left=False)
                ax.grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
                ax.grid(axis='y', visible=False)

                ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]),rango[j][1]+0.05*(rango[j][1]-rango[j][0]))
            
        plt.show()

        return


    def plot_distancia_intercluster(num_clusters, centers, size, colors):
        fig = plt.figure(figsize=(5,5))
        mpl.style.use('default')

        mds = MDS(random_state=SEED)
        centers_mds = mds.fit_transform(centers)

        plt.scatter(centers_mds[:,0], centers_mds[:,1], s=size**1.6, alpha=0.75, c=colors)
        
        for i in range(num_clusters):
            plt.annotate(str(i+1), xy=centers_mds[i], va='center', ha='center')
            
        xl,xr = plt.xlim()
        yl,yr = plt.ylim()

        plt.xlim(xl-(xr-xl)*0.13,xr+(xr-xl)*0.13)
        plt.ylim(yl-(yr-yl)*0.13,yr+(yr-yl)*0.13)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        return


    def plot_parallel_coordinates(X, metric_SC_samples, num_clusters, clusters, attributes, colors):
        plt.figure()
        mpl.style.use('default')

        X = X.assign(cluster=clusters)
        # X.loc[:, 'cluster'] = clusters
        X.loc[:,'SC'] = metric_SC_samples
        df = X

        # si se desea aclarar la figura, se pueden eliminar los objetos más lejanos, es decir, SC < umbral, p.ej., 0.3
        df = df.loc[df['SC']>=0.3]

        colors_parcoor = [(round((i//2)/num_clusters+(1/num_clusters)*(i%2),3),'rgb'+str(colors[j//2])) for i,j in zip(range(2*num_clusters),range(2*num_clusters))]

        fig = px.parallel_coordinates(df, dimensions=attributes,
                                    color="cluster", range_color=[-0.5, num_clusters-0.5],
                                    color_continuous_scale=colors_parcoor)

        fig.update_layout(coloraxis_colorbar=dict(
            title="Clusters",
            tickvals=[i for i in range(num_clusters)],
            ticktext=["Cluster "+str(i+1) for i in range(num_clusters)],
            lenmode="pixels", len=500,
        ))
        
        fig.show()

        return fig


    def apply_meanshift(X, X_normalizado):
        t = time.time()

        cluster_model = MeanShift()
        clusters_predicted = cluster_model.fit_predict(X_normalizado)

        tiempo = time.time() - t
        print("Tempo de execução: {:.2f} segundos".format(tiempo)+'\n')

        centers = pd.DataFrame(cluster_model.cluster_centers_,columns=list(X))
        clusters = pd.DataFrame(clusters_predicted, index=X.index, columns=['cluster'])
        X_clustering = pd.concat([X, clusters], axis=1)

        print("Tamanho de cada cluster:")
        size = clusters['cluster'].value_counts()
        for num, i in size.items():
            print('%s: %5d (%5.2f%%)' % (num+1, i, 100*i/len(clusters)))

        size = size.sort_index()
        print()

        return X_clustering, clusters_predicted, clusters, centers, size


    def apply_aglomerative(X, X_normalizado, num_clusters):
        t = time.time()

        cluster_model = AgglomerativeClustering(n_clusters=num_clusters)
        clusters_predicted = cluster_model.fit_predict(X_normalizado)

        ex_time = time.time() - t
        print("Tempo de execução: {:.2f} segundos".format(ex_time))

        clusters = pd.DataFrame(clusters_predicted, index=X.index, columns=['cluster'])
        X_clustering = pd.concat([X, clusters], axis=1)

        return  X_clustering, clusters_predicted, clusters


    def apply_kmodes(X, num_clusters):
        t = time.time()

        kmodes_huang = KModes(n_clusters=num_clusters, init='Huang', verbose=0)
        clusters_predicted = kmodes_huang.fit(X)

        ex_time = time.time() - t
        print("Tempo de execução: {:.2f} segundos".format(ex_time))

        centers = pd.DataFrame(kmodes_huang.cluster_centroids_,columns=list(X))
        clusters = pd.DataFrame(clusters_predicted, index=X.index, columns=['cluster'])
        X_clustering = pd.concat([X, clusters], axis=1)

        print("\nTamanho de cada cluster:")
        size = clusters['cluster'].value_counts()
        for num, i in size.items():
            print('%s: %5d (%5.2f%%)' % (num, i, 100*i/len(clusters)))

        size = size.sort_index()

        return  X_clustering, clusters_predicted, clusters, centers, size

    # - https://datascience.stackexchange.com/questions/22/k-means-clustering-for-mixed-numeric-and-categorical-data
    # - https://cse.hkust.edu.hk/~qyang/Teaching/537/Papers/huang98extensions.pdf
    # - https://pypi.org/project/kmodes/

    def apply_kmeans(X, X_normalizado, num_clusters):
        t = time.time()

        k_means = KMeans(init='k-means++', n_clusters=num_clusters, n_init=5, random_state=SEED)
        clusters_predicted = k_means.fit_predict(X_normalizado)

        tiempo = time.time() - t
        print("Tempo de execução: {:.2f} segundos".format(tiempo))

        centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
        clusters = pd.DataFrame(clusters_predicted, index=X.index, columns=['cluster'])

        X_clustering = pd.concat([X, clusters], axis=1)

        print("\nTamanho de cada cluster:")
        size = clusters['cluster'].value_counts()
        for num, i in size.items():
            print('%s: %5d (%5.2f%%)' % (num, i, 100*i/len(clusters)))

        size = size.sort_index()

        return  X_clustering, clusters_predicted, clusters, centers, size
    

    def apply_gaussianmixture(X, X_normalizado, num_clusters):
        t = time.time()

        cluster_model = GaussianMixture(n_components=num_clusters, random_state=SEED)
        clusters_predicted = cluster_model.fit_predict(X_normalizado)

        tiempo = time.time() - t
        print("Tempo de execução: {:.2f} segundos".format(tiempo))

        clusters = pd.DataFrame(clusters_predicted, index=X.index, columns=['cluster'])
        X_clustering = pd.concat([X, clusters], axis=1)

        print("Tamanho de cada cluster:")
        size = clusters['cluster'].value_counts()
        for num, i in size.items():
            print('%s: %5d (%5.2f%%)' % (num, i, 100*i/len(clusters)))

        size = size.sort_index()

        return  X_clustering, clusters_predicted, clusters, size
    

    def apply_spectral(X, X_normalizado, num_clusters):
        t = time.time()

        cluster_model = SpectralClustering(n_clusters=num_clusters, random_state=SEED)
        clusters_predicted = cluster_model.fit_predict(X_normalizado)

        tiempo = time.time() - t
        print("Tiempo de ejecución: {:.2f} segundos".format(tiempo))

        clusters = pd.DataFrame(clusters_predicted, index=X.index, columns=['cluster'])
        X_clustering = pd.concat([X, clusters], axis=1)

        return X_clustering, clusters_predicted, clusters