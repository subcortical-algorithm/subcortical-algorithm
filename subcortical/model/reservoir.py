import numpy as np
import matplotlib.pyplot as plt
import torch
import copy


def toCUDA(np_array):
    return torch.from_numpy(np_array.astype(np.float32)).cuda()


def reservoir_topology(N, rho_scale, tau, dt, p_con):
    J = np.random.randn(N, N).astype(np.float32)
    p_mar = np.random.rand(N, N).astype(np.float32)
    p_mar = p_mar < p_con
    J = J * p_mar
    numx = 1 - dt/tau
    M = dt/tau * J + numx * np.eye(N).astype(np.float32)
    rho_all, _ = np.linalg.eig(M)
    rho = max(np.abs(rho_all))

    J = J / (rho - numx) * (rho_scale - numx)

    return J.astype(np.float32)


def reservoir_topology_lowrank(N, p_con):
    J = np.random.randn(N, N).astype(np.float32)
    p_mar = np.random.rand(N, N).astype(np.float32)
    p_mar = p_mar < p_con
    J = J * p_mar
    J_std = np.std(J)
    J = J / J_std / np.sqrt(N)
    return J.astype(np.float32)



class ReservoirNet(object):
    def __init__(self, n_layer, spatial_dim=None, kernel_sigma=None, 
                 generate_weights=True, diffusive_coupling=False,
                 noise_level=0, stp_connection=False, low_rank=False,
                 n_class=None):
        # ============= Parameters =============
        self.n_class         = n_class
        self.batch_size      = 1
        self.layernum        = n_layer
        self.dt              = 1
        self.N               = [130] * self.layernum
        # input dimensions
        self.N0              = (30, 40)
        self.p_in            = [0.05] * self.layernum
        self.strength_in     = [1] * self.layernum
        # feedforward input matrix
        self.Win             = [None] * self.layernum
        self.p_con           = [0.3] * self.layernum
        # recurrent connection matrix
        self.J               = [np.zeros([x, x]) for x in self.N]
        self.rho_scale       = [1.2] * self.layernum
        self.tau             = [2.5, 12.5, 25]
        self.t               = 0
        self.diffusive_coupling = diffusive_coupling
        self.spatial_dim     = spatial_dim
        self.kernel_sigma    = kernel_sigma
        self.noise_level     = noise_level
        self.stp_connection  = stp_connection
        self.x_stp           = [toCUDA(np.ones([self.batch_size, x])) for x in self.N]
        self.u_stp           = [toCUDA(np.zeros([self.batch_size, x])) for x in self.N]
        self.u_stp_history   = [[] for x in range(self.layernum)]
        self.x_stp_history   = [[] for x in range(self.layernum)]
        self.tau_stf         = 500
        self.tau_std         = 1000
        self.incre_stf       = 0.1
        self.lowrank_flag    = low_rank
        
        if generate_weights:
            self.__generate_reservoir__()


    def __generate_reservoir__(self):
        if self.diffusive_coupling:
            if self.kernel_sigma is None:
                raise Exception("Attribute 'kernel_sigma' needs to be specified if diffusive_coupling option is enabled.")
            if self.spatial_dim is None:
                raise Exception("Attribute 'spatial_dim' needs to be specified if diffusive_coupling option is enabled.")
            for layer in range(self.n_layer):
                assert self.N[layer]==np.prod(self.spatial_dim[layer]), "Spatial dimension configuration is not" \
                                                                "compatible with the number of neurons in layer %d." % layer

        self.x_r            = [toCUDA(np.zeros([self.batch_size, self.N[x]])) for x in range(self.layernum)]
        self.x_mean = [[] for x in range(self.layernum)]
        self.x_all  = None
        self.x_stp           = [toCUDA(np.ones([self.batch_size, x])) for x in self.N]
        self.u_stp           = [toCUDA(np.zeros([self.batch_size, x])) for x in self.N]
        

        if self.lowrank_flag:

            for l in range(self.layernum):
                self.J[l] = reservoir_topology_lowrank(self.N[l], self.p_con[l])

                low_rank = np.matmul(np.random.randn(self.N[l], self.n_class), np.random.randn(self.n_class, self.N[l])) / self.N[l]
                self.J[l] = (100*self.J[l] + 100*low_rank / self.N[l])*1
        else:
            for l in range(self.layernum):
                self.J[l] = reservoir_topology(self.N[l], self.rho_scale[l],
                                               self.tau[l], self.dt, self.p_con[l])

        for l in range(self.layernum):
            self.J[l] = toCUDA(self.J[l])

        if self.diffusive_coupling:
            for layer in range(self.layernum):
                if layer == 0:
                    h_pre, w_pre = self.N0
                else:
                    h_pre, w_pre = self.spatial_dim[layer-1]

                h, w = self.spatial_dim[layer]

                self.Win[layer] = torch.nn.Conv2d(in_channels=1, out_channels=h*w, kernel_size=(h_pre, w_pre), bias=False)

                x = np.linspace(0, w_pre, w)
                y = np.linspace(0, h_pre, h)

                ind = 0
                for i_x in x:
                    for i_y in y:
                        # https://pytorch.org/docs/stable/nn.html#conv2d
                        self.Win[layer].weight.data[ind, 0, :, :] = self.__generate_diffusive_kernel__(i_x, i_y, h_pre, w_pre, layer)
                        ind = ind + 1

                self.Win[layer] = self.Win[layer].cuda()
                self.Win[layer].weight.requires_grad = False

        else:
            for layer in range(self.layernum):
                if layer == 0:
                    self.p_matrix   = np.random.rand(np.prod(self.N0), self.N[layer]) < self.p_in[layer]
                    self.Win[layer] = toCUDA((np.random.rand(np.prod(self.N0), self.N[layer]).astype(np.float32) * 2 - 1) 
                                                * self.p_matrix * self.strength_in[layer])
                    
                else:
                    self.p_matrix   = np.random.rand(self.N[layer - 1], self.N[layer]) < self.p_in[layer]
                    self.Win[layer] = toCUDA((np.random.rand(self.N[layer - 1], self.N[layer]).astype(np.float32) * 2 - 1) 
                                                * self.p_matrix * self.strength_in[layer])


    def __generate_diffusive_kernel__(self, x, y, h_pre, w_pre, layer, cutoff=0.1):
        f = lambda x,y: np.exp(-((x-mu_x)**2/(2*sig_x**2)+(y-mu_y)**2/(2*sig_y**2)))
        sig_x = w_pre * self.kernel_sigma[layer]
        sig_y = h_pre * self.kernel_sigma[layer]
        mu_x, mu_y = int(x), int(y)
        _x = np.arange(w_pre)
        _y = np.arange(h_pre)
        _x, _y = np.meshgrid(_x, _y)
        filter = f(_x, _y) > cutoff
        kernel = (np.random.rand(h_pre, w_pre)-0.5) * 2
        kernel = kernel * filter

        return toCUDA(kernel)

    def __dynamics_layer__(self, inputs, x_r):
        with torch.no_grad():
            for layer in range(self.layernum):

                if layer == 0:
                    f_input = self.__feedforward_input__(inputs, self.Win[layer], layer=layer) 
                else:
                    f_input = self.__feedforward_input__(np.tanh(x_r[layer-1]), self.Win[layer], layer=layer)

                r_input = torch.matmul(torch.tanh(x_r[layer]), self.J[layer])

                # Short-term plasticity dynamics
                noise = self.__noise__(layer)

                dx = -x_r[layer] + self.u_stp[layer]*self.x_stp[layer]*r_input + f_input + noise
                x_r[layer] = x_r[layer] + dx / self.tau[layer] * self.dt

                if self.stp_connection:
                    du_stp = -self.u_stp[layer]/self.tau_stf + self.incre_stf*(1-self.u_stp[layer])*torch.tanh(x_r[layer])
                    self.u_stp[layer] += du_stp * self.dt
                    dx_stp = (1-self.x_stp[layer])/self.tau_std - self.u_stp[layer]*self.x_stp[layer]*torch.tanh(x_r[layer])
                    self.x_stp[layer] += dx_stp * self.dt
                    
                    self.u_stp_history[layer].append(torch.mean(self.u_stp[layer], axis=1).cpu().numpy())
                    self.x_stp_history[layer].append(torch.mean(self.x_stp[layer], axis=1).cpu().numpy())


        return x_r

    def __feedforward_input__(self, activity, feedforward_weight, layer):
        if self.diffusive_coupling:
            if layer == 0:
                spatial_dim = self.N0
            else:
                spatial_dim = self.spatial_dim[layer-1]

            return feedforward_weight(activity.reshape(activity.shape[0], 1, *spatial_dim)).reshape(activity.shape[0], -1)
        else:
            return torch.matmul(activity, self.Win[layer])

    def __noise__(self, layer):
        if self.noise_level == 0:
            return 0
           
        noise = self.noise_level * np.random.randn(self.batch_size, self.N[layer])
        return toCUDA(noise)

    def forward(self, inputs):
        self.inputs = inputs
        self.x_r = self.__dynamics_layer__(inputs, self.x_r)

        for l in range(self.layernum):
            self.x_mean[l].append(torch.mean(torch.abs(self.x_r[l]), 1))

        self.x_all = torch.cat(self.x_r, dim=1)

        self.t += 1

    def reset(self):
        self.x_r = [toCUDA(np.zeros([self.batch_size, self.N[x]])) for x in range(self.layernum)]
        self.x_mean =[[] for x in range(self.layernum)]
        self.x_all  = None
        self.t = 0
        self.u_stp_history = [[] for x in range(self.layernum)]
        self.x_stp_history = [[] for x in range(self.layernum)]
        self.x_stp           = [toCUDA(np.ones([self.batch_size, x])) for x in self.N]
        self.u_stp           = [toCUDA(np.zeros([self.batch_size, x])) for x in self.N]


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.reset()

    def visualize_activity(self):
        color_map = ['#b6fcd5', '#c6e2ff', '#faebd7', '#dcedc1']
        n_layer = len(self.x_r)
        batch_size = self.batch_size

        for b in range(batch_size):
            for i in range(n_layer):
                x_r = self.x_r[i].cpu().numpy()[b, :]
                n, bins, patches = plt.hist(x=x_r, bins=30, color=color_map[i], alpha=0.7, 
                                            rwidth=0.85, label="Layer %d"%i, density=True)
                _total = len(x_r)
                _in_range = np.sum(-2.5 < k < 2.5 for k in x_r)
                _percent = _in_range / _total * 100

                plt.text(-10, 0.75-0.1*i, "Layer %d: %.2f%% in [-2.5, 2.5] " % (i+1, _percent))

            plt.grid(True)
            plt.title('Reservoir Activity Histogram')
            plt.legend()
            xx = np.linspace(-10, 10, 100)
            plt.plot(xx, np.tanh(xx), linewidth=2)
            plt.show()

    def visualize_stp(self):
        n_layer = len(self.x_r)
        batch_size = self.batch_size
        u_stp_history_np = np.array(self.u_stp_history).transpose(2,0,1)
        x_stp_history_np = np.array(self.x_stp_history).transpose(2,0,1)

        for b in range(batch_size):
            for l in range(n_layer):
                plt.subplot(n_layer, n_layer, (b*n_layer)+l+1)
                if b == 0:
                    plt.title("Layer %d" %(l+1))
                plt.plot(u_stp_history_np[b,l,:], label="u", linewidth=2)
                plt.plot(x_stp_history_np[b,l,:], label="x", linewidth=2)
                plt.legend()
                plt.grid(True)
        plt.show()



class ReservoirConfig:
    def __init__(self,config):
        self.config = config
        self.__parse_config__()

    def __parse_config__(self):
        n_layer = self.config["n_layer"]
        self._reservoir = ReservoirNet(n_layer=n_layer, generate_weights=False)
        self._reservoir.__dict__.update(self.config)
        self._reservoir.__generate_reservoir__()
        self.__dict__.update(self._reservoir.__dict__)

    def forward(self, inputs):
        self._reservoir.forward(inputs)
        self.__dict__.update(self._reservoir.__dict__)

    def reset(self):
        self._reservoir.reset()
        self.__dict__.update(self._reservoir.__dict__)

    def set_batch_size(self, batch_size):
        self._reservoir.set_batch_size(batch_size)
        self.__dict__.update(self._reservoir.__dict__)

    def visualize_activity(self):
        self._reservoir.visualize_activity()
        self.__dict__.update(self._reservoir.__dict__)

    def visualize_stp(self):
        self._reservoir.visualize_stp()
        self.__dict__.update(self._reservoir.__dict__)        




