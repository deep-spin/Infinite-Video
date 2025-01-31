# coding: utf-8
"""
Attention modules
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.distributions as dist

from entmax import Sparsemax, Entmax15, EntmaxBisect
from .basis_functions import (PowerBasisFunctions,
                                     SineBasisFunctions,
                                     CosineBasisFunctions,
                                     GaussianBasisFunctions,
                                     RetangularBasisFunctions)
from .continuous_sparsemax import ContinuousSparsemax
from .continuous_softmax import ContinuousSoftmax

import math

import numpy as np

import pickle
import matplotlib.pyplot as plt



class LongTermAttention(nn.Module):
    def __init__(self, head_size:int , length: int, target_len:int,  attn_func: str, attn_num_basis: int,
                  continuous: bool, attn_drop: float, infinite_memory: bool, n_layers: int,
                  n_heads: int, affines: bool, mask: bool, mask_type: str, kl_regularizer: bool, proj_key, proj_value, sigma_0, mu_0, sticky_memories, sigmas, tau, **kwargs):

        super(LongTermAttention, self).__init__()

        self.device = 'cuda'
        self.length = length #memory length
        self.target_len = target_len #target length / transformer length
        self.head_size = head_size
        self.attn_num_basis = attn_num_basis
        self.continuous = continuous # whether attention over memory vectors is continuous
        self.attn_func = attn_func # normalizing function
        self.n_head = n_heads
        self.sigmas = sigmas
        self.kl_regularizer = kl_regularizer
        self.sigma_0 = sigma_0
        self.mu_0 = mu_0
        self.proj_key = proj_key
        self.proj_value = proj_value
        
        self.affines=affines # whether mu, sigma should be computed using affine transformations


        self.sticky_memories=sticky_memories

        self.mem_threshold=2048
        self.infinite_memory = infinite_memory # whether the memory is infinite

        self.nb_samples=512 # number of samples used for update
        self.tau = tau #compressing factor
        self.count = 0

        self.x_past=None # previous memory vectors
        self.B_past=None # previous coefficient matrix

        self.ridge_penalty=0.5 # ridge penalty
        self.padding = True

        self.spacing='linear'

    def get_basis(self, length, target_len):
        def compute_G(l, psi, positions, padding=True):

            F = torch.zeros(self.attn_num_basis, positions.size(0))

            basis_functions = psi
            F[:, :] = basis_functions.evaluate(positions.unsqueeze(1)).t()

            I = torch.eye(self.attn_num_basis)
            G = F.t().matmul((F.matmul(F.t()) + self.ridge_penalty * I).inverse())

            if padding:
                if l % 2:
                    G = G[((l-1)//2):(-(l-1)//2), :]
                else:
                    G = G[(l//2):-(l//2), :]

            return G.to(self.device)
        padding = self.padding
        attn_func=self.attn_func
        attn_num_basis = self.attn_num_basis
        if self.continuous:

            self.psi=[None]
            self.Gs=[None for _ in range(length+1)]
            lengths=[]
            for i in range(length):
                self.psi.append([])
                if (i+1)%target_len==0:
                    lengths.append(i+1)
            if length not in lengths:
                lengths.append(length)
            for l in lengths:
                # get positions for memory vectors
                self.add_retangular_basis_functions(self.psi[l], attn_num_basis, device=self.device)

                if self.spacing=='linear':
                    if padding:
                        if l % 2:
                            shift = 1 / float(l)
                            positions = torch.linspace(-.5+shift, 1.5-shift, 2*l-1).to(self.device)
                        else:
                            shift = 1 / float(2*l)
                            positions = torch.linspace(-.5+shift, 1.5-shift, 2*l).to(self.device)
                    else:
                        shift = 1 / float(2*l)
                        positions = torch.linspace(shift, 1-shift, l).to(self.device)
                elif self.spacing=='log':
                    if padding:
                        if l % 2:
                            shift = 1 / float(l)
                            positions = torch.linspace(-.5+shift, 1.5-shift, 2*l-1).to(self.device)
                        else:
                            shift = 1 / float(2*l)
                            positions = torch.linspace(-.5+shift, 1.5-shift, 2*l).to(self.device)

                        pos = np.e**(np.log(1+1)*torch.arange(1,length+1)/length)-1
                        positions = torch.cat([positions[:int(l/2)],pos.to(self.device),positions[-int(l/2):]])

                    else:
                        positions = np.e**(np.log(1+1)*torch.arange(1,length+1)/length)-1
            
                # compute basis functions
                self.Gs[l]=compute_G(l, self.psi[l][0], positions, padding=padding) # [L,N]
                self.positions = positions[int(l/2):-int(l/2)]

            # compute samples for memory update
            if self.infinite_memory:
                tm_tau = torch.arange(1,self.nb_samples+1).float()
                tm_l = torch.arange(self.nb_samples+1,length+self.nb_samples+1).float()
                tm_tau = tm_tau*self.tau/self.nb_samples # positions of old vectors
                tm_l = self.tau + (1-self.tau)*(tm_l-self.nb_samples)/length # positions of new vectors
                positions_inf = torch.cat([tm_tau, tm_l],0).to(self.device) # positions

                if padding:
                    if l % 2:
                        shift = 1 / float(length+self.nb_samples)
                        positions_pad = torch.linspace(-.5+shift, 1.5-shift, 2*(length+self.nb_samples)-1).to(self.device)
                    else:
                        shift = 1 / float(2*length+self.nb_samples)
                        positions_pad = torch.linspace(-.5+shift, 1.5-shift, 2*(length+self.nb_samples)).to(self.device)
                    positions_pad_ = torch.FloatTensor([i for i in positions_pad if i<0]).to(self.device)
                    positions_pad__ = torch.FloatTensor([i for i in positions_pad if i>1]).to(self.device)
                    positions_inf = torch.cat([positions_pad_,positions_inf,positions_pad__], dim=0)

                self.samples=None
                for t in tm_tau:
                    if self.samples is None:
                        self.samples = self.psi[l][0].evaluate(t/self.tau)
                    else:
                        self.samples = torch.cat([self.samples,self.psi[l][0].evaluate(t/self.tau)], dim=0)

                # compute G for the infinite case
                self.G_inf = compute_G(self.nb_samples+length, self.psi[l][0], positions_inf, padding=padding) #[L+nb_samples,N]

                if self.sticky_memories:
                    self.bins = torch.linspace(0,1,129).to(device=self.device) #self.positions
                    self.nb_bins_cat=1
                    self.bins_cat = dist.Categorical(torch.ones(self.nb_bins_cat))

    def add_gaussian_basis_functions(self, psi, nb_basis, sigmas, device):
        mu, sigma = torch.meshgrid(torch.linspace(0, 1, nb_basis // len(sigmas)), torch.Tensor(sigmas))
        mu = mu.flatten().to(device)
        sigma = sigma.flatten().to(device)
        self.basis_mu=mu
        self.basis_sigma=sigma
        assert mu.size(0) == nb_basis
        psi.append(GaussianBasisFunctions(mu=mu, sigma=sigma))

    def add_retangular_basis_functions(self, psi, nb_basis, device):
        width = torch.ones(nb_basis, device=device) / nb_basis
    
        # Compute the centers (midpoints) of each bin
        edges = torch.linspace(0, 1, nb_basis + 1, device=device)
        mu = (edges[:-1] + edges[1:]) / 2 
        psi.append(RetangularBasisFunctions(mu=mu, sigma=width))

    def value_function(self, x, inf=False):
        if inf:
            G = self.G_inf # [nb_sample+L,N]
        else:
            G = self.Gs[x.size(-1)] # [L,N]
        B = torch.matmul(x, G) # [B,e,N]
        B = B.permute(0,2,1) # [B,N,e]
        
        return B

    def update_inf(self, x):
        if self.B_past is not None:       
            if self.sticky_memories:
                bins = self.bins.clone()
                bins[0]=-.000001
                bins[-1]=1.000001
                prob_density = self.compute_probability(self.score, t=bins)
                cum_prob = torch.cumulative_trapezoid(prob_density, bins, dim=-1).to(self.device)
                p = (cum_prob[..., 1:] - cum_prob[..., :-1]).sum(dim=(1, 2))
                p = p / p.sum(-1, keepdim=True)  # Normalize over the last dimension (bins)
                p = dist.Categorical(p)
                b = p.sample((self.nb_samples,))
                t = self.bins_cat.sample((self.nb_samples, 1)).to(device=self.device)
                ts = (t*(self.bins[b+1]-self.bins[b])/self.nb_bins_cat +self.bins[b]).transpose(1,0)
                samples = self.psi[self.length][0].batch_evaluate(ts[0]).contiguous()

                xm_tau = self.B_past.transpose(-1,-2).matmul(samples.transpose(-1,-2)) # [B,e,nb_samples]
            else:
                xm_tau = self.B_past.transpose(-1,-2).matmul(self.samples.transpose(-1,-2)) # [B,e,nb_samples]
                
            
            x = torch.cat([xm_tau,x], dim=2) # [B,e,nb_samples+L]
            B = self.value_function(x, inf=True) # [B,N,e]
        else:
            B = self.value_function(x)
        
        self.B_past=B.detach()
        self.x_past=x
        return B
    
    def score(self, t):
        psis = self.psis[0].batch_evaluate(t)
        query = self.queries/ (self.d_head ** 0.5) # divide by sqrt(d_head) [B,h,q,d]
        keys = self.keys.transpose(-1, -2)
        keys = torch.matmul(keys, psis.T) #[B,h,d,1]
        scores = torch.matmul(query, keys) #[B,h,q,1] 
        return scores

    def compute_probability(self, score_fn, num_points=1000, t=None):
        """
        Compute probability distribution p(t).
        
        Args:
            score_fn (callable): Function that computes z(t)
            num_points (int): Number of points for numerical integration
        
        Returns:
            tuple: (probabilities, normalization constant)
        """
        if t is None:
            # Create integration points
            t = torch.linspace(0, 1, num_points).to(self.device)

        scores = score_fn(t)
        prob = torch.exp(scores) / torch.trapz(torch.exp(scores), t, dim=-1).unsqueeze(-1)
        return prob
    
    def expected_value(self, score_fn, num_points=1000):
        """
        Compute expected value E_p[V(t)] using nested integration.
        
        Args:
            score_fn (callable): Function that computes z(t)
            value_fn (callable): Function that computes v(t)
            num_points (int): Number of points for numerical integration
        
        Returns:
            torch.Tensor: Expected value
        """
        # Create integration points
        t = torch.linspace(0, 1, num_points).to(self.device)
        
        # Compute basis functions
        self.psis = []
        self.add_retangular_basis_functions(self.psis, self.attn_num_basis, self.device)
        psi = self.psis[0].batch_evaluate(t)
        # Compute probability distribution
        prob = self.compute_probability(score_fn, num_points)
        # Compute values at integration points
        values = self.values
        # Compute p(t) * psi(t)
        # Reshape psi for broadcasting to match the shape of prob
        psi_broadcasted = psi.unsqueeze(1).unsqueeze(2).unsqueeze(3) 

        # Expand psi to match the dimensions of prob (num_points, batch_size, n_head, qlen, 256)
        psi_broadcasted = psi_broadcasted.expand(num_points, self.batch_size, self.n_head, self.qlen, self.attn_num_basis)
        integrand = torch.matmul(prob.permute(3,0,1,2).unsqueeze(-1).unsqueeze(-1), psi_broadcasted.unsqueeze(-2)).permute(1, 2, 3, 4, 5, 0).squeeze(-3)

        integral  = torch.trapz(integrand, t, dim=-1)
        # Matrix multiply with values
        expected_value = torch.matmul(integral, values)  # [B, h, q, d]
        
        return expected_value
    
    def forward(self, k, q, new_doc, layer_n):
        self.device = k.device
        if self.continuous:
            klen = int(k.size(1)/32)
            self.length = klen
            batch_size = k.size(0) #batch size
            qlen = q.size(1) #query length
            self.qlen = qlen
            self.batch_size = batch_size
            self.d_head = self.head_size #head size
            self.get_basis(klen, klen)
            # clean memory if going through different document
            if new_doc:
                self.B_past=None 
                self.x_past=None
            
            k = k.reshape(batch_size, klen, 32, 768).mean(dim=2)
            k = k.transpose(1,2)
            # perform memory update
            if self.infinite_memory:
                B = self.update_inf(k)
            else: # compute input continuous approximation
                B = self.value_function(k) # [B,N,e]
            
            keys = self.proj_key(B)
            values = self.proj_value(B)
            query = q
            self.queries = query.view(batch_size,qlen,self.n_head,self.d_head).transpose(1,2) # [B,h,q,d]
            self.keys = keys.view(batch_size,self.attn_num_basis,self.n_head,self.d_head).transpose(1,2) # [B,h,N,d]
            self.values = values.view(batch_size,self.attn_num_basis,self.n_head,self.d_head).transpose(1,2) # [B, h, q, N]
            context = self.expected_value(self.score)  # Shape [1, 32, 768]
            
            output_density = True
            if output_density:
                try:
                    if self.alphas_save:
                        aux=True
                except:
                    self.alphas_save=[]
                import math
                t = torch.linspace(0, 0.25, 256).to(self.device)
                density1 = self.compute_probability(self.score, num_points=11, t=t)
                t = torch.linspace(0.25, 0.5, 256).to(self.device)
                density2 = self.compute_probability(self.score, num_points=11, t=t)
                t = torch.linspace(0.5, 1, 256).to(self.device)
                density3 = self.compute_probability(self.score, num_points=11, t=t)
                density = torch.cat((density1, density2, density3), dim=-1)
                alphas = density / torch.sum(density, dim=-1).unsqueeze(-1)
                #density = torch.clamp(density, min=1e-6)
                alphas =alphas.permute(2,0,1, 3).cpu() 
                density = self.compute_probability(self.score, num_points=2048, t=None)
                alphas1 = density / torch.sum(density, dim=-1).unsqueeze(-1)
                #density = torch.clamp(density, min=1e-6)
                alphas1 =alphas1.permute(2,0,1, 3).cpu() 
                with open('./alphas_uniform','wb') as f:
                    pickle.dump(alphas,f)
                #with open('./alphas_total_8_chunks_tau_0.9_uniform','wb') as f:
                #    pickle.dump(alphas1,f)
            return context.contiguous().transpose(1,2).reshape(1, qlen, -1)
        
