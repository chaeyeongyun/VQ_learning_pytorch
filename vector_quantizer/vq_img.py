import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class CosinesimCodebook(nn.Module):
    def __init__(self):
        super().__init__()


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        kmeans_init,
        kmeans_iter,
        decay,
        eps
        ):
        super().__init__()
        self.decay = decay
        if kmeans_init:
            raise NotImplementedError
        else:
            self.embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
            self.embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
    def forward(self, x):
        x = x.float()
        x_shape, dtype = x.shape, x.dtype
        flatten_x = rearrange(x, 'b ... c -> b (...) d')
        # TODO: flatten과 embed의 거리를 구하는 부분. torch.cdist와 그냥 식으로 구현하는 방법 어떻게 할지 생각 좀 해봐야 될 것 같다
        embed_index = 

codebook_dict = {'euclidean':EuclideanCodebook,
                 'cosine':CosinesimCodebook,}

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim,
        decay=0.8,
        eps=1e-5,
        kmeans_init=False,
        kmeans_iters=10,
        distance='euclidean',
        commitment_weight = 1,  
        ):
        super().__init__()
        codebook_dim = codebook_dim if codebook_dim!=None else dim
        self.codebook_size = codebook_size
        # TODO: projection require
        self.eps = eps
        self.commitment_weight = commitment_weight
        codebook_class = codebook_dict[distance]
        self.codebook = codebook_class(
            dim=codebook_dim,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init
            kmeans_iters = kmeans_iters,
            decay = decay,
            eps = eps
        )
    def forward(self, x):
        x_shape, device = x.shape, x.device
        x_h, x_w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c') # (B, HxW, C)
        quantize, embed_index = self.codebook(x)
        loss = torch.Tensor([0.], device=device, requires_grad=self.training)
        if self.training:
            #TODO: 이거 안해주면 생기는 오류 보기
            quantize = x + (quantize - x).detach() # quantize 가져오는 과정에서 gradient가 끊겨서 x에 어떤 상수를 더해주는 방식으로 해서 gradient가 이어지도록 만든 것 같다
            if self.commitment_weight > 0:
                detached_quantize = quantize.detach()
                commitment_loss = F.mse_loss(detached_quantize, x)
                loss += commitment_loss * self.commitment_weight
            
        quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=x_h, w=x_w)
        embed_index = rearrange(embed_index, 'b (h w) ... -> b h w ...', h=x_h, w=x_w)
        return quantize, embed_index, loss