import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# PCA
from sklearn.decomposition import PCA

iris = load_iris()
pca = PCA(n_components=3)
iris_trans = pca.fit_transform(iris.data)
iris_pca_df = pd.DataFrame(iris_trans)
iris_pca_df['species'] = iris.target
fig = px.scatter_3d(iris_pca_df, x=0, y=1, z=2, color='species')
st.plotly_chart(fig, use_container_width=False, sharing="streamlit")