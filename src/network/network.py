import torch
import torch.nn as nn
from pdb import set_trace as stx

# 从 render 函数里面的 run_network 函数查找输入样例
class DensityNetwork(nn.Module):
    def __init__(self, encoder, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1, last_activation="sigmoid"):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.encoder = encoder
        self.in_dim = encoder.output_dim + 16
        self.bound = bound
        
        self.embed = nn.Linear(1, 16)

        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips 
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers-1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        self.activations = nn.ModuleList([nn.LeakyReLU() for i in range(0, num_layers-1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        # stx()
        '''
            input: (N_rays x N_samples, 4)
            经过encoder后变成: (N_rays x N_samples, 32)
        '''
        pos = x[..., :3]
        pos = self.encoder(pos, self.bound)     # encoder 把 x 从低维变成高维
        rho = x[..., -1].unsqueeze(-1)
        rho = self.embed(rho)
        x = torch.cat((pos, rho), dim=-1)

        input_pts = x[..., :self.in_dim]    # 就是x

        for i in range(len(self.layers)):

            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)
        
        return x

class DensityNetwork_debug(nn.Module):
    def __init__(self, bound=0.2, num_layers=8, hidden_dim=256, skips=[4], out_dim=1, last_activation="sigmoid"):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.bound = bound
        self.in_dim = 32
        
        # Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) if i not in skips 
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim) for i in range(1, num_layers-1, 1)])
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        # Activations
        self.activations = nn.ModuleList([nn.LeakyReLU() for i in range(0, num_layers-1, 1)])
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        else:
            raise NotImplementedError("Unknown last activation")

    def forward(self, x):
        # stx()
        '''
            input: (N_rays x N_samples, 3)
            经过encoder后变成: (N_rays x N_samples, 32) - (1024 * 192, 32)
        '''
        
        input_pts = x    # 就是x

        for i in range(len(self.layers)):

            linear = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], -1)

            x = linear(x)
            x = activation(x)
        
        return x

'''
    配置文件中的实例参数
    network:
        net_type: mlp
        num_layers: 4
        hidden_dim: 32
        skips: [2]
        out_dim: 1
        last_activation: sigmoid
        bound: 0.3
'''

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    import sys
    sys.path.append('/home/lzhou/SAX-NeRF-rho')
    from src.encoder import get_encoder

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = get_encoder("hashgrid", input_dim=3,
                num_levels=16, level_dim=2,
                multires=6, 
                degree=4,
                 base_resolution=16, log2_hashmap_size=19)
    
    model = DensityNetwork(encoder, num_layers=4,hidden_dim=32,skips=[2],out_dim=1,last_activation="sigmoid",bound=0.3).to(device)
    
    # inputs = torch.randn((1024*192, 32)).to(device)
    inputs = torch.rand(1024*192, 4) * 0.6 - 0.3
    inputs = inputs.to(device)
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')