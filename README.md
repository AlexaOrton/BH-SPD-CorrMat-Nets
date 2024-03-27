# SPDNet, SPDNetBN \& U-SPDNet for Market Regime Detection Using JSE Top 60 and Synthetic Block Hierarchical Correlation Matrices 
We perform market regime detection by testing three deep representation learning models tailored to the SPD Riemannian manifold of correlation matrices constructed from Bloomberg JSE Top 60 traded stock price returns data and synthetically-generated block hierarchical correlation matrices.

The synthetic nested price return process is an adaptation/parameterised version of the general Hierarchically Nested Factor Model (HNFM)  from Multivariate Data (Tumminello et al., 2007) https://doi.org/10.1209/0295-5075/78/30006 and expanded on in Agglomerative Likelihood Clustering by Yelibi and Gebbie (2021) (https://doi.org/10.1088/1742-5468/ac3661 with code repo: https://github.com/lyelibi/timeseries_generator).

The basic architecture of the SPDNet with Riemannian batch normalisation (RBN) is informed by the work of Brooks et al. (2019) in Second-Order Networks in pytorch (https://doi.org/10.1007/978-3-030-26980-7_78; http://arxiv.org/abs/1909.02414). Schwander (2021)'s formulation of the NN-optimisation-functional form of the iterative problem-solving approach is found at https://gitlab.lip6.fr/schwander/torchspdnet. 

The architecture of the U-SPDNet is found at: https://github.com/GitWR/U-SPDNet based on the work in U-SPDNet: An SPD manifold learning-based neural network for visual classification by Wang et al. (2022) (https://www.sciencedirect.com/science/article/pii/S0893608022004713).
