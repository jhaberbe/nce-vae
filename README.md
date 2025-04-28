Simple VAE using Noise Contrastive Estimation in the interior. 

This is the first rough output, but you can check out the notebook and see that it works reasonably well. Definitely would recommend training on GPU, but CPU can still give feasible results in <1h. 

There is also a VAE that uses a gaussian mixture model to regularize the latent dimension, as opposed to the typical KL divergence. It also works reasonably well on 3k PBMC, though larger training samples have not been tested yet. 
