import itertools
from functools import partial
from typing import Any, Mapping, Tuple, Callable, Optional
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk


from utils import resnet_femnist
from utils import resnet_cifar
from utils import resnet_cifar100
from utils import hk_transformer
from utils import hk_initializers
from utils import hk_mlp


from deq.solver.fpi import FixedPointIteration
from deq.solver.anderson import AndersonAcceleration
from deq.solver.rootfinding_wrapper import BroydenRootFinding

from deq.solver.linear_solve import solve_gmres, solve_normal_cg
from deq.solver.tree_util import tree_add, tree_sub, tree_l2_norm
from deq.solver.deq_seq import deq

# Default Batch Normalization.
_DEFAULT_BN_CONFIG = {
  'decay_rate': 0.9,
  'eps': 1e-5,
  'create_scale': True,
  'create_offset': True
}

# MLP Default Hyperparameters.
he_normal = hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')
_DEFAULT_ACTIVATION_FN = jax.nn.relu
_DEFAULT_HIDDEN_DIM = 512

# Transformer Default Hyperparameters.
NUM_LAYERS = 4
NUM_HEADS = 4 # Number of attention heads.
MODEL_SIZE = 64
KEY_SIZE = 32
DROPOUT_RATE = 0.1

##################################################
################### GET MODEL ####################
##################################################

def get_model(model, dataset, forward_solver = None, backward_solver = None):

    # Fixed-Point Iteration or Root-Finding
    fpi = True
    # Solver used for implicit differentiation (backward pass).
    if backward_solver == "normal_cg":
        implicit_solver = partial(solve_normal_cg, tol=1e-3, maxiter=50)
    elif backward_solver == "gmres":
        implicit_solver = partial(solve_gmres, tol=1e-3, maxiter=50)

    if forward_solver == "anderson":
        fpi = True
        fixed_point_solver = partial(AndersonAcceleration,
                                    history_size=5,
                                    ridge=1e-4,
                                    maxiter=20,
                                    tol=1e-3, implicit_diff=True,
                                    implicit_diff_solve=implicit_solver)
    elif forward_solver == "broyden":
        fpi = False
        fixed_point_solver = partial(BroydenRootFinding,
                                        tol = 1e-3,
                                        implicit_diff_solve=implicit_solver)                                 
    else:
        fpi = True
        fixed_point_solver = partial(FixedPointIteration,
                                    maxiter=20,
                                    tol=1e-2, implicit_diff=True,
                                    implicit_diff_solve=implicit_solver)

    # Model Selection     
    #######################
    # Deep Neural Network #
    #######################                                  
    if model == "dnn_3l":
        model_prefix = "dnn_3l"
        last_layers = "player"
                
        if dataset == "femnist":
            return partial(dnn_3l,
                           hidden_dims = _DEFAULT_HIDDEN_DIM,
                           num_classes = 62,
                           activation = _DEFAULT_ACTIVATION_FN), model_prefix, last_layers   
        elif dataset == "cifar10":
            return partial(dnn_3l, 
                           num_classes = 10, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM * 2, 
                           activation = _DEFAULT_ACTIVATION_FN), model_prefix, last_layers
        elif dataset == "cifar100":
            return partial(dnn_3l, 
                           num_classes = 100, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM * 2, 
                           activation = _DEFAULT_ACTIVATION_FN), model_prefix, last_layers
                                             
    elif model == "dnn_5l":
        model_prefix = "dnn_5l"
        last_layers = "player"

        if dataset == "femnist":
            return partial(dnn_5l,
                           hidden_dims = _DEFAULT_HIDDEN_DIM,
                           num_classes = 62,
                           activation = _DEFAULT_ACTIVATION_FN), model_prefix, last_layers   
        elif dataset == "cifar10":
            return partial(dnn_5l, 
                           num_classes = 10, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM * 2, 
                           activation = _DEFAULT_ACTIVATION_FN), model_prefix, last_layers
        elif dataset == "cifar100":
            return partial(dnn_5l, 
                           num_classes = 100, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM * 2, 
                           activation = _DEFAULT_ACTIVATION_FN), model_prefix, last_layers
                                        
        
    #######################
    ######## Resnet #######
    #######################     
                      
    elif model == "resnet34":

        model_prefix = "resnet34"
        last_layers = "player"

        if dataset == "femnist":
            return partial(resnet34_femnist, 
                           num_classes = 62), model_prefix, last_layers
        elif dataset == "cifar10":
            return partial(resnet34_cifar, 
                           num_classes = 10), model_prefix, last_layers
        elif dataset == "cifar" or dataset == "cifar100":
            return partial(resnet34_cifar, 
                           num_classes = 100), model_prefix, last_layers      
        
    elif model == "resnet20":

        model_prefix = "resnet20"
        last_layers = "player"

        if dataset == "femnist":
            return partial(resnet20_femnist, 
                           num_classes = 62), model_prefix, last_layers
        elif dataset == "cifar10":
            return partial(resnet20_cifar, 
                           num_classes = 10), model_prefix, last_layers
        elif dataset == "cifar100":
            return partial(resnet20_cifar, 
                           num_classes = 100), model_prefix, last_layers
   
    elif model == "resnet14":

        model_prefix = "resnet14"
        last_layers = "player"

        if dataset == "femnist":
            return partial(resnet14_femnist, 
                           num_classes = 62), model_prefix, last_layers
        elif dataset == "cifar10":
            return partial(resnet14_cifar, 
                           num_classes = 10), model_prefix, last_layers
        elif dataset == "cifar100":
            return partial(resnet14_cifar, 
                           num_classes = 100), model_prefix, last_layers


    ############################
    ######## Transformer #######
    ############################ 
    elif model == "transformer4":
        model_prefix = "transfomer4"
        last_layers = "player"
        return partial(transformer_shakespeare,
                       num_layers=4, 
                       num_heads = 4,
                       model_size = 64), model_prefix, last_layers

    elif model == "transformer8":
        model_prefix = "transfomer8"
        last_layers = "player"
        return partial(transformer_shakespeare,
                       num_layers=8, 
                       num_heads = 4,
                       model_size = 64), model_prefix, last_layers
    
    elif model == "transformer12":
        model_prefix = "transfomer12"
        last_layers = "player"
        return partial(transformer_shakespeare,
                       num_layers=12,
                       num_heads = 4,
                       model_size = 64), model_prefix, last_layers
    
    #############################
    ########### DEQ-MLP #########
    #############################   
    elif model == "deq_mlp":

        print(f'[INFO] Forward/Backward: {forward_solver}/{backward_solver}')
        model_prefix = "deq_mlp"
        last_layers = "player"

        def deqmlp_ra_v1(inputs, num_classes = 62, hidden_dim = 512, is_training = True):
            deqmlp = DEQ_MLP_RA_V1(fixed_point_solver,
                                   num_classes = num_classes,
                                   hidden_dim = hidden_dim,
                                   activation = _DEFAULT_ACTIVATION_FN,
                                   fpi = fpi)
            return deqmlp(inputs, is_training)

        if dataset == "femnist":
            #return fedjax.models.emnist.create_conv_model()
            return partial(deqmlp_ra_v1, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM, 
                           num_classes = 62), model_prefix, last_layers   
        elif dataset == "cifar10":
            #return fedjax.models.emnist.create_conv_model()
            return partial(deqmlp_ra_v1, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM * 2, 
                           num_classes = 10), model_prefix, last_layers 
        elif dataset == "cifar100":
            #return fedjax.models.emnist.create_conv_model()
            return partial(deqmlp_ra_v1, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM * 2, 
                           num_classes = 100), model_prefix, last_layers 
                  
    elif model == "deq_mlp_v2":

        print(f'[INFO] Forward/Backward: {forward_solver}/{backward_solver}')
        model_prefix = "deq_mlp_v2"
        last_layers = "player"

        def deqmlp_ra_v2(x, num_classes = 62, hidden_dim = 512, is_training = True):
            deqmlp = DEQ_MLP_RA_V2(fixed_point_solver,
                        num_classes = num_classes,
                        hidden_dim = hidden_dim,
                        activation = _DEFAULT_ACTIVATION_FN,
                        fpi = fpi)
            return deqmlp(x, is_training)

        if dataset == "femnist":
            #return fedjax.models.emnist.create_conv_model()
            return partial(deqmlp_ra_v2, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM, 
                           num_classes = 62), model_prefix, last_layers  
        elif dataset == "cifar10":
            #return fedjax.models.emnist.create_conv_model()
            return partial(deqmlp_ra_v2, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM * 2, 
                           num_classes = 10), model_prefix, last_layers 
        elif dataset == "cifar100":
            #return fedjax.models.emnist.create_conv_model()
            return partial(deqmlp_ra_v2, 
                           hidden_dim = _DEFAULT_HIDDEN_DIM * 2, 
                           num_classes = 100), model_prefix, last_layers 
                          
    ##########################
    ####### DEQ-Resnet #######
    ##########################  

    elif model == "deq_resnet_s":

        print(f'[INFO] Forward/Backward: {forward_solver}/{backward_solver}')
        model_prefix = "deq__resnet_s"
        last_layers = "player"

        def DEQResnet_S(x, 
                        channel_1 = 64, channel_2 = 128, 
                        linear_1 = 256, linear_2 = 128, 
                        num_classes = 62, is_training = True):
            
            deqresnet = DEQ_Resnet_Softplus(fixed_point_solver, 
                                            channel_1, channel_2, 
                                            linear_1, linear_2, 
                                            num_classes, fpi = fpi)
            return deqresnet(x, is_training)
                
        if dataset == "femnist":
            return partial(DEQResnet_S, 
                           num_classes = 62), model_prefix, last_layers  
              
        elif dataset == "cifar10":
            return partial(DEQResnet_S,
                           num_classes = 10), model_prefix, last_layers
        
        elif dataset == "cifar100":
            return partial(DEQResnet_S, 
                           num_classes = 100), model_prefix, last_layers      

    elif model == "deq_resnet_m":

        print(f'[INFO] Forward/Backward: {forward_solver}/{backward_solver}')
        model_prefix = "deq__resnet_m"
        last_layers = "player"

        def DEQResnet_M(x, 
                        channel_1 = 128, channel_2 = 256, 
                        linear_1 = 256, linear_2 = 128, 
                        num_classes = 62, is_training = True):
            
            deqresnet = DEQ_Resnet_Softplus(fixed_point_solver, 
                                            channel_1, channel_2, 
                                            linear_1, linear_2, 
                                            num_classes, fpi = fpi)
            return deqresnet(x, is_training)
                
        if dataset == "femnist":
            return partial(DEQResnet_M, 
                           num_classes = 62), model_prefix, last_layers  
              
        elif dataset == "cifar10":
            return partial(DEQResnet_M,
                           num_classes = 10), model_prefix, last_layers
        
        elif dataset == "cifar100":
            return partial(DEQResnet_M, 
                           num_classes = 100), model_prefix, last_layers    

    elif model == "deq_resnet_relu":

        print(f'[INFO] Forward/Backward: {forward_solver}/{backward_solver}')
        model_prefix = "deq__resnet_relu"
        last_layers = "player"

        def DEQResnet_ReLU(x, channel_1 = 32, channel_2 = 64, 
                                 linear_1 = 256, linear_2 = 128, 
                                 num_classes = 62, is_training = True):
            deqresnet = DEQ_Resnet_ReLU(fixed_point_solver, 
                                        channel_1, channel_2, 
                                        linear_1, linear_2, 
                                        num_classes, fpi = fpi)
            return deqresnet(x, is_training)
                
        if dataset == "femnist":
            return partial(DEQResnet_ReLU, 
                           channel_1 = 64, channel_2 = 128, 
                           num_classes = 62), model_prefix, last_layers  
              
        elif dataset == "cifar10":
            return partial(DEQResnet_ReLU,
                           channel_1 = 128, channel_2 = 256,
                           linear_1 = 256, linear_2 = 128, 
                           num_classes = 10), model_prefix, last_layers
        
        elif dataset == "cifar100":
            return partial(DEQResnet_ReLU, 
                           channel_1 = 128, channel_2 = 256, 
                           linear_1 = 256, linear_2 = 128, 
                           num_classes = 100), model_prefix, last_layers    
        
    elif model == "deq_resnet_softplus":

        print(f'[INFO] Forward/Backward: {forward_solver}/{backward_solver}')
        model_prefix = "deq__resnet_softplus"
        last_layers = "player"

        def DEQResnet_Softplus(x, channel_1 = 32, channel_2 = 64, 
                                linear_1 = 256, linear_2 = 128, 
                                num_classes = 62, is_training = True):
            deqresnet = DEQ_Resnet_Softplus(fixed_point_solver, 
                                            channel_1, channel_2, 
                                            linear_1, linear_2, 
                                            num_classes, fpi = fpi)
            return deqresnet(x, is_training)
                
        if dataset == "femnist":
            return partial(DEQResnet_Softplus, 
                           channel_1 = 64, 
                           channel_2 = 128, 
                           num_classes = 62), model_prefix, last_layers  
              
        elif dataset == "cifar10":
            return partial(DEQResnet_Softplus,
                           channel_1 = 128, channel_2 = 256,
                           linear_1 = 256, linear_2 = 128, 
                           num_classes = 10), model_prefix, last_layers
        
        elif dataset == "cifar100":
            return partial(DEQResnet_Softplus, 
                           channel_1 = 128, channel_2 = 256, 
                           linear_1 = 256, linear_2 = 128, 
                           num_classes = 100), model_prefix, last_layers    
    
    #####################################
    ########### DEQ-Transformer #########
    ##################################### 
    elif model == "deq_transformer":
        print(f'[INFO] Forward/Backward: {forward_solver}/{backward_solver}')
        model_prefix = "deq_transformer"
        last_layers = "player"

        if dataset == "shakespeare":

            deq_shakespeare = deq_transformer(num_layers = 4,  
                                              num_heads = 4, 
                                              d_model = 64,
                                              vocab_size = 90, 
                                              dropout_rate = 0.1)
            
            return deq_shakespeare, model_prefix, last_layers


################################################
################  DEQ-MLP Haiku ################
################################################
# For completeness, we also allow Anderson acceleration for solving
# the implicit differentiation linear system occurring in the backward pass.

def solve_linear_system_fixed_point(matvec, v):
  """Solve linear system matvec(u) = v.

  The solution u* of the system is the fixed point of:
    T(u) = matvec(u) + u - v
  """
  def fixed_point_fun(u):
    d_1_T_transpose_u = tree_add(matvec(u), u)
    return tree_sub(d_1_T_transpose_u, v)

  aa = AndersonAcceleration(fixed_point_fun,
                            history_size=5, tol=1e-2,
                            ridge=1e-4, maxiter=20)
  return aa.run(v)[0]


class DEQMLPFixedPoint(hk.Module):
    """Batched computation of ``block`` using ``fixed_point_solver``."""

    #block: Any  # nn.Module
    #fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration

    def __init__(self, block, fixed_point_solver):
        super().__init__()
        self.block = block
        self.fixed_point_solver = fixed_point_solver
        self.rng = jax.random.PRNGKey(42)

    def __call__(self, x):
        # shape of a single example
        # lift params
        block_params = hk.experimental.lift(self.block.init, name="DEQMLP_Block")(self.rng, x[0], x[0])        
        #block_params = self.block.init(self.rng, x[0], x[0])
        #block_params = self.param("block_params", init, x)

        def block_apply(z, x, block_params):
            return self.block.apply(block_params, self.rng, z, x)

        solver = self.fixed_point_solver(fixed_point_fun=block_apply)
        def batch_run(x, block_params):
            return solver.run(x, x, block_params)[0]

        # We use vmap since we want to compute the fixed point separately for each
        # example in the batch.
        return jax.vmap(batch_run, in_axes=(0,None), out_axes=0)(x, block_params)

class DEQMLPFixedPointRA(hk.Module):
    """Batched computation of ``block`` using ``fixed_point_solver``."""

    #block: Any  # nn.Module
    #fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration

    def __init__(self, block, fixed_point_solver):
        super().__init__()
        self.block = block
        self.fixed_point_solver = fixed_point_solver
        self.rng = jax.random.PRNGKey(42)

    def __call__(self, z_init, x):
        # shape of a single example
        # lift params
        if z_init == None:
            z_init = x

        block_params = hk.experimental.lift(self.block.init, 
                                            name="DEQMLP_Block")(self.rng, z_init[0], x[0])        
        #block_params = self.block.init(self.rng, x[0], x[0])
        #block_params = self.param("block_params", init, x)

        def block_apply(z, x, block_params):
            return self.block.apply(block_params, self.rng, z, x)

        solver = self.fixed_point_solver(fixed_point_fun=block_apply)
        def batch_run(z_init, x, block_params):
            return solver.run(z_init, x, block_params)[0]

        # We use vmap since we want to compute the fixed point separately for each
        # example in the batch.
        return jax.vmap(batch_run, in_axes=(0,0,None), out_axes=0)(z_init, x, block_params)

class MLPBlock(hk.Module):

    
    def __init__(self,
                 init_scale: float = 0.25,
                 widening_factor: int = 3,
                 hiddens: int = 100,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
                 with_bias: bool  = False,
                 name: Optional[str] = None):

        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self._hiddens = hiddens
        self._activation = activation
        self._bias = with_bias
        
    def __call__(self, z, x):
        # Note that stddev=0.01 is important to avoid divergence.
        # Empirically it ensures that fixed point iterations converge.
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        #z = hk.Flatten()(z)
        # Defining conv1 channel = 154
        #y = hk.Linear(self._widening_factor * self._hiddens, w_init=initializer)(z)
        # Applying relu for self.conv1(z)
        #y = self._activation(y)
        # Applying Group Norm for nn.relu(self.conv1(z))
        #h = hk.Flatten()(z)
        h = hk.Linear(self._hiddens, w_init=initializer, with_bias=self._bias)(z)
        h = h + x
        h = self._activation(h) 
        #Cx = hk.Linear(...)(x)
        #Bz = hk.Linear(..., disable bias)(z)
        #Cx + Bz
        # Defining conv2 ouput_channel = 28
        # Combining input with layer                      
        # Applying Group Norm for inputs + self.conv2(y)
        return h

class MLPBlock_V1(hk.Module):

    
    def __init__(self,
                 init_scale: float = 0.25,
                 widening_factor: int = 1,
                 hiddens: int = 128,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
                 with_bias: bool  = False,
                 name: Optional[str] = None):

        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self._hiddens = hiddens
        self._activation = activation
        self._bias = with_bias
        
    def __call__(self, z, x):
        # Note that stddev=0.01 is important to avoid divergence.
        # Empirically it ensures that fixed point iterations converge.
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        #z = hk.Flatten()(z)
        h = hk.Linear(self._widening_factor * self._hiddens,
                      with_bias=self._bias,
                      name="linear_z")(z)
        #h = self._activation(h)
        #h = hk.Linear(self._hiddens, 
        #              with_bias=self._bias)(h)
        h = h + x
        h = self._activation(h) 
        return h


class MLPBlock_V2(hk.Module):

    
    def __init__(self,
                 init_scale: float = 0.25,
                 widening_factor: int = 2,
                 hiddens: int = 128,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
                 with_bias: bool  = False,
                 name: Optional[str] = None):

        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self._hiddens = hiddens
        self._activation = activation
        self._bias = with_bias
        
    def __call__(self, z, x):
        # Note that stddev=0.01 is important to avoid divergence.
        # Empirically it ensures that fixed point iterations converge.
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        #z = hk.Flatten()(z)
        h = hk.Linear(self._widening_factor * self._hiddens,
                      with_bias=self._bias, 
                      w_init=initializer)(z)
        h = self._activation(h)
        h = hk.Linear(self._hiddens, 
                      w_init=initializer, 
                      with_bias=self._bias)(h)
        h = h + x
        h = self._activation(h) 
        return h

class MLPBlock_V3(hk.Module):

    def __init__(self,
                 init_scale: float = 0.25,
                 hiddens: int = 1024,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
                 with_bias: bool  = False,
                 name: Optional[str] = None):

        super().__init__(name=name)
        self._init_scale = init_scale
        #self._widening_factor = widening_factor
        self._hiddens = hiddens
        self._activation = activation
        self._initializer = None
        #self._initializer = hk.initializers.VarianceScaling(self._init_scale)
        self._bias = with_bias
        
    def __call__(self, z, x):
        # Note that stddev=0.01 is important to avoid divergence.
        # x -> Cx.
        h = hk.Linear(self._hiddens,
                      w_init=self._initializer,
                      with_bias=self._bias,
                      name = "linear_z")(z)
        h = h + x
        h = self._activation(h) 
        return h

class MLPBlock_Softplus(hk.Module):

    def __init__(self,
                 init_scale: float = 0.25,
                 hiddens: int = 1024,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.softplus,
                 with_bias: bool  = False,
                 name: Optional[str] = None):

        super().__init__(name=name)
        self._init_scale = init_scale
        #self._widening_factor = widening_factor
        self._hiddens = hiddens
        self._activation = activation
        self._initializer = None
        #self._initializer = hk.initializers.VarianceScaling(self._init_scale)
        self._bias = with_bias
        
    def __call__(self, z, x):
        # Note that stddev=0.01 is important to avoid divergence.
        # x -> Cx.
        h = hk.Linear(self._hiddens,
                      w_init=self._initializer,
                      with_bias=self._bias,
                      name = "linear_z")(z)
        """
        h = self._activation(h) 
        h = hk.Linear(self._hiddens,
                      w_init=self._initializer,
                      with_bias=self._bias,
                      name = "linear_z")(z)     
        """   
        h = h + x
        h = self._activation(h) 
        return h
        

class DEQ_MLP_RA_V1(hk.Module):

    def __init__(self, 
                 solver, 
                 hidden_dim: int = 128,
                 init_scale: float = 0.25,
                 num_classes: int = 62,
                 fpi: bool = True,
                 with_bias = False,
                 activation = jax.nn.relu,
                 name: Optional[str] = None):
                 
        super().__init__(name=name)
        self._solver = solver
        self._hiddens = hidden_dim
        self._fpi = fpi
        self._num_classes = num_classes
        self._init_scale = init_scale
        self._bias = with_bias
        self._z = None
        self._activation = activation

    def __call__(self, x, is_training = True):

        initializer = hk.initializers.VarianceScaling(self._init_scale)
        y = hk.Flatten()(x)
        y = hk.Linear(self._hiddens,
                      with_bias = self._bias,
                      #w_init=initializer
                      name="linear_x")(y)       
        
        def MLP(z, x):
            mlp = MLPBlock_V1(hiddens = self._hiddens,
                              activation = self._activation)
            if self._fpi:
                #print("Fixed-Point Iteration")           o
                return mlp(z, x)
            else:
                #print("Root-Finding") 
                return mlp(z, x) - z

        block = hk.transform(MLP)

        deq_fixed_point = DEQMLPFixedPointRA(block, self._solver)

        y = deq_fixed_point(self._z, y)
        self._z = y

        y = hk.Flatten()(y)
        y = hk.Linear(128)(y)
        embedding = self._activation(y) 
        y = hk.Linear(self._num_classes, name="player_logits")(embedding)

        return y, embedding

class DEQ_MLP_RA_V2(hk.Module):

    def __init__(self, 
                 solver, 
                 hidden_dim: int = 128,
                 init_scale: float = 0.25,
                 num_classes: int = 62,
                 fpi: bool = True,
                 with_bias = False,
                 activation = jax.nn.relu,
                 name: Optional[str] = None):
                 
        super().__init__(name=name)
        self._solver = solver
        self._hiddens = hidden_dim
        self._fpi = fpi
        self._num_classes = num_classes
        self._init_scale = init_scale
        self._bias = with_bias
        self._z = None
        self._activation = activation

    def __call__(self, x, is_training = True):

        initializer = hk.initializers.VarianceScaling(self._init_scale)
        y = hk.Flatten()(x)
        y = hk.Linear(self._hiddens,
                      with_bias = self._bias,
                      #w_init=initializer
                      name="linear_x")(y)       
        #y = self._activation(y) 
        
        def MLP(z, x):
            mlp = MLPBlock_V1(hiddens = self._hiddens,
                              activation = self._activation)
            if self._fpi:
                return mlp(z, x)
            else:
                return mlp(z, x) - z

        block = hk.transform(MLP)

        deq_fixed_point = DEQMLPFixedPointRA(block, self._solver)

        y = deq_fixed_point(self._z, y)
        self._z = y

        y = hk.BatchNorm(**_DEFAULT_BN_CONFIG)(y, is_training)
        #y = y.reshape((x.shape[0], -1)) # flatten
        #y = hk.Flatten()(y)
        y = hk.Linear(self._hiddens)(y)
        embedding = self._activation(y) 
        y = hk.Linear(self._num_classes, name="player_logits")(embedding)

        return y, embedding

       
################################################
################  DEQ-RESNET Haiku #############
################################################
class ResnetBlock(hk.Module):
    """Adapted from haiku module for CNN with dropout to match LEAF model.
    
    This must be defined as a custom hk.Module because only a single positional
    argument is allowed when using hk.Sequential.
    """
    
    def __init__(self, channels, output_channels,
                 num_groups = 8, kernel_shape = (3,3), bias = False):
        super().__init__()
        self._channels = channels
        self._output_channels = output_channels
        self._num_groups = num_groups
        self._kernel_shape = kernel_shape
        self._bias = bias
        self._init = hk.initializers.RandomNormal(stddev=0.01)
        
    def __call__(self, z, x):
        # Note that stddev=0.01 is important to avoid divergence.
        # Empirically it ensures that fixed point iterations converge.
        y = z
        # Defining conv1 channel = 154
        y = hk.Conv2D(output_channels=self._output_channels,
                      kernel_shape= self._kernel_shape, 
                      padding='SAME', with_bias = self._bias,
                      w_init=self._init, b_init=self._init)(y)
        # Applying relu for self.conv1(z)
        y = jax.nn.relu(y)
        # Applying Group Norm for nn.relu(self.conv1(z))
        y = hk.GroupNorm(groups = self._num_groups)(y)
        # Defining conv2 ouput_channel = 28
        y = hk.Conv2D(output_channels=self._channels,
                      kernel_shape= self._kernel_shape, 
                      padding='SAME', with_bias = self._bias,
                      w_init=self._init, b_init=self._init)(y)
        # Combining input with layer                      
        y = y + x
        # Applying Group Norm for inputs + self.conv2(y)
        y = hk.GroupNorm(groups = self._num_groups)(y)  
        # group_norm3(nn.relu(z + self.group_norm2(inputs + self.conv2(y))))        
        y = y + z      
        y = jax.nn.relu(y)        
        y = hk.GroupNorm(groups = self._num_groups)(y)  

        return y

class ResnetBlock_Softplus(hk.Module):
    """Adapted from haiku module for CNN with dropout to match LEAF model.
    
    This must be defined as a custom hk.Module because only a single positional
    argument is allowed when using hk.Sequential.
    """
    
    def __init__(self, channels, output_channels,
                 num_groups = 8, kernel_shape = (3,3), bias = False):
        super().__init__()
        self._channels = channels
        self._output_channels = output_channels
        self._num_groups = num_groups
        self._kernel_shape = kernel_shape
        self._bias = bias
        self._init = hk.initializers.RandomNormal(stddev=0.01)
        
    def __call__(self, z, x):
        # Note that stddev=0.01 is important to avoid divergence.
        # Empirically it ensures that fixed point iterations converge.
        y = z
        # Defining conv1 channel = 154
        y = hk.Conv2D(output_channels=self._output_channels,
                      kernel_shape= self._kernel_shape, 
                      padding='SAME', with_bias = self._bias,
                      w_init=self._init, b_init=self._init)(y)
        # Applying relu for self.conv1(z)
        y = jax.nn.softplus(y)
        # Applying Group Norm for nn.relu(self.conv1(z))
        y = hk.GroupNorm(groups = self._num_groups)(y)
        # Defining conv2 ouput_channel = 28
        y = hk.Conv2D(output_channels=self._channels,
                      kernel_shape= self._kernel_shape, 
                      padding='SAME', with_bias = self._bias,
                      w_init=self._init, b_init=self._init)(y)
        # Combining input with layer                      
        y = y + x
        # Applying Group Norm for inputs + self.conv2(y)
        y = hk.GroupNorm(groups = self._num_groups)(y)  
        # group_norm3(nn.relu(z + self.group_norm2(inputs + self.conv2(y))))        
        y = y + z      
        y = jax.nn.softplus(y)        
        y = hk.GroupNorm(groups = self._num_groups)(y)  

        return y

class DEQFixedPoint(hk.Module):
    """Batched computation of ``block`` using ``fixed_point_solver``."""

    #block: Any  # nn.Module
    #fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration

    def __init__(self, block, fixed_point_solver):
        super().__init__()
        self.block = block
        self.fixed_point_solver = fixed_point_solver
        self.rng = jax.random.PRNGKey(42)

    def __call__(self, x):
        # shape of a single example
        # lift params
        block_params = hk.experimental.lift(self.block.init, name="DEQResnet_Block")(self.rng, x[0], x[0])        
        #block_params = self.block.init(self.rng, x[0], x[0])
        #block_params = self.param("block_params", init, x)

        def block_apply(z, x, block_params):
            return self.block.apply(block_params, self.rng, z, x)

        solver = self.fixed_point_solver(fixed_point_fun=block_apply)
        def batch_run(x, block_params):
            return solver.run(x, x, block_params)[0]

        # We use vmap since we want to compute the fixed point separately for each
        # example in the batch.
        return jax.vmap(batch_run, in_axes=(0,None), out_axes=0)(x, block_params)
    
class DEQFixedPointRA(hk.Module):
    """Batched computation of ``block`` using ``fixed_point_solver`` with warm-start."""

    #block: Any  # nn.Module
    #fixed_point_solver: Any  # AndersonAcceleration or FixedPointIteration

    def __init__(self, block, fixed_point_solver):
        super().__init__()
        self.block = block
        self.fixed_point_solver = fixed_point_solver
        self.rng = jax.random.PRNGKey(42)

    def __call__(self, z_init, x):
       
        if z_init == None:
            z_init = x
        # lift params
        block_params = hk.experimental.lift(self.block.init,
                                            name="DEQResnet_Block")(self.rng,
                                                                    z_init[0],
                                                                    x[0])        
        def block_apply(z, x, block_params):
            return self.block.apply(block_params, self.rng, z, x)

        solver = self.fixed_point_solver(fixed_point_fun=block_apply)
        def batch_run(z_init, x, block_params):
            return solver.run(z_init, x, block_params)[0]

        # We use vmap since we want to compute the fixed point separately for each
        # example in the batch.
        return jax.vmap(batch_run, in_axes=(0,0,None), out_axes=0)(z_init, x, block_params)

class DEQ_Resnet_ReLU(hk.Module):
    """ 
    Deep Equilibrium Resnet Model
    """
    def __init__(self, solver, channels, output_channels,
                 num_classes = 10, kernel_shape = (3,3), bias = False, fpi = True):
        super().__init__()
        self._solver = solver
        self._num_classes = num_classes
        self._channels = channels
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._bias = bias
        self._init = hk.initializers.RandomNormal(stddev=0.01)
        self._fpi = fpi
        self._z = None

    def __call__(self, x, is_training = True):

        y = hk.Conv2D(output_channels=self._channels,
                      kernel_shape= self._kernel_shape, 
                      padding='SAME', with_bias = self._bias,
                      w_init=self._init, b_init=self._init)(x)
        y = jax.nn.relu(y)
        y = hk.BatchNorm(**_DEFAULT_BN_CONFIG)(y, is_training)
        y = hk.MaxPool(
                window_shape=(1, 3, 3, 1), # equivalent to pool_size=[2, 2] of tf
                strides=(1, 2, 2, 1), # equivalent to strides=2 of tf
                padding='SAME')(y)
        
        def ResNet(z, x):
            resnet = ResnetBlock(self._channels, self._output_channels)
            if self._fpi:
                #print("Fixed-Point Iteration")           o
                return resnet(z, x)
            else:
                #print("Root-Finding") 
                return resnet(z, x) - z

        block = hk.transform(ResNet)

        deq_fixed_point = DEQFixedPointRA(block, self._solver)

        y = deq_fixed_point(self._z, y)
        self._z = y

        y = hk.MaxPool(
                window_shape=(1, 3, 3, 1), # equivalent to pool_size=[2, 2] of tf
                strides=(1, 2, 2, 1), # equivalent to strides=2 of tf
                padding='SAME')(y)
        y = hk.Flatten()(y)

        y = hk.Linear(256)(y)
        y = jax.nn.relu(y) 
        #y = hk.BatchNorm(**_DEFAULT_BN_CONFIG)(y, is_training)
        
        y = hk.Linear(128)(y)
        embedding = jax.nn.relu(y) 
        y = hk.BatchNorm(**_DEFAULT_BN_CONFIG)(embedding, is_training)
        
        y = hk.Linear(self._num_classes, name="player_logits")(y)

        return y, embedding

class DEQ_Resnet_Softplus(hk.Module):
    """ 
    Deep Equilibrium Resnet Model
    """
    def __init__(self, solver, channels, output_channels, linear1 = 128, linear2 = 256,
                 num_classes = 10, kernel_shape = (3,3), bias = False, fpi = True):
        super().__init__()
        self._solver = solver
        self._num_classes = num_classes
        self._channels = channels
        self._output_channels = output_channels
        self._linear1 = linear1
        self._linear2 = linear2
        self._kernel_shape = kernel_shape
        self._bias = bias
        self._init = hk.initializers.RandomNormal(stddev=0.01)
        self._fpi = fpi
        self._z = None

    def __call__(self, x, is_training = True):

        y = hk.Conv2D(output_channels=self._channels,
                      kernel_shape= self._kernel_shape, 
                      padding='SAME', with_bias = self._bias,
                      w_init=self._init, b_init=self._init)(x)
        y = jax.nn.softplus(y)
        y = hk.BatchNorm(**_DEFAULT_BN_CONFIG)(y, is_training)
        
    
        y = hk.MaxPool(
                window_shape=(1, 3, 3, 1), # equivalent to pool_size=[2, 2] of tf
                strides=(1, 2, 2, 1), # equivalent to strides=2 of tf
                padding='SAME')(y)
        
        def ResNet(z, x):
            resnet = ResnetBlock_Softplus(self._channels, self._output_channels)
            if self._fpi:
                #print("Fixed-Point Iteration")           o
                return resnet(z, x)
            else:
                #print("Root-Finding") 
                return resnet(z, x) - z

        block = hk.transform(ResNet)

        deq_fixed_point = DEQFixedPointRA(block, self._solver)

        y = deq_fixed_point(self._z, y)
        self._z = y

        y = hk.MaxPool(
                window_shape=(1, 3, 3, 1), # equivalent to pool_size=[2, 2] of tf
                strides=(1, 2, 2, 1), # equivalent to strides=2 of tf
                padding='SAME')(y)
        
        #y = jnp.mean(y, axis=[1, 2])

        y = hk.Flatten()(y)

        y = hk.Linear(self._linear1)(y)
        y = jax.nn.softplus(y) 
        #y = hk.BatchNorm(**_DEFAULT_BN_CONFIG)(y, is_training)
        
        y = hk.Linear(self._linear2)(y)
        embedding = jax.nn.softplus(y) 
        y = hk.BatchNorm(**_DEFAULT_BN_CONFIG)(embedding, is_training)
        
        y = hk.Linear(self._num_classes, name="player_logits")(y)

        return y, embedding
 
#######################################
################  MLP     #############
#######################################

def dnn_3l(inputs, hidden_dims = 256, num_classes = 10, activation = jax.nn.relu):

  return hk.Sequential([
    hk.Flatten(),
    hk.nets.MLP([hidden_dims,
                 hidden_dims,
                 128,
                 num_classes], activation = activation)
  ])(inputs), num_classes


def dnn_5l(inputs, hidden_dims = 256, num_classes = 10, activation = jax.nn.relu):

  return hk.Sequential([
    hk.Flatten(),
    hk.nets.MLP([hidden_dims,
                 hidden_dims,
                 hidden_dims,
                 hidden_dims,
                 128,
                 num_classes], activation = activation)
  ])(inputs), num_classes

def dnn_10l(inputs, hidden_dims = 256, num_classes = 10, activation = jax.nn.relu):

  return hk.Sequential([
    hk.Flatten(),
    hk.nets.MLP([hidden_dims,
                 hidden_dims,
                 hidden_dims,
                 hidden_dims,
                 hidden_dims,
                 hidden_dims,
                 hidden_dims,
                 hidden_dims,
                 hidden_dims,
                 128,
                 num_classes], activation = activation)
  ])(inputs), num_classes

#############################################
################  RESNET 34  ################
#############################################
def resnet34_femnist(data, num_classes=62, is_training = True, **kwarg):

  net = resnet_femnist.ResNet34(num_classes,
                                resnet_v2=False,
                                bn_config={'decay_rate': 0.9})
  return net(data, is_training)

def resnet34_cifar(data, num_classes = 10, is_training = True):
    if num_classes == 10:
        net = resnet_cifar.ResNet34(num_classes,
                                    bn_config=_DEFAULT_BN_CONFIG)
    else: 
        net = resnet_cifar100.ResNet34(num_classes,
                                       bn_config=_DEFAULT_BN_CONFIG)
    return net(data, is_training)


#############################################
################  RESNET 20  ################
#############################################
def resnet20_femnist(data, num_classes = 62, is_training = True):
  net = resnet_femnist.ResNet20(num_classes,
                                resnet_v2=False,
                                bn_config=_DEFAULT_BN_CONFIG)
  return net(data, is_training)

def resnet20_cifar(data, num_classes = 10, is_training = True):
    if num_classes == 10:
        net = resnet_cifar.ResNet20(num_classes,
                                    bn_config=_DEFAULT_BN_CONFIG)
    else: 
        net = resnet_cifar100.ResNet20(num_classes,
                                    bn_config=_DEFAULT_BN_CONFIG)
    return net(data, is_training)

#############################################
################  RESNET 14  ################
#############################################
def resnet14_femnist(data, num_classes = 62, is_training = True):
  """Forward application of the resnet."""
  net = resnet_femnist.ResNet14(num_classes,
                                resnet_v2=False,
                                bn_config=_DEFAULT_BN_CONFIG)
  return net(data, is_training)

def resnet14_cifar(data, num_classes = 10, is_training = True):
    """Forward application of the resnet."""
    if num_classes == 10:
        net = resnet_cifar.ResNet14(num_classes,
                                    bn_config=_DEFAULT_BN_CONFIG)
    else: 
        net = resnet_cifar100.ResNet14(num_classes,
                                    bn_config=_DEFAULT_BN_CONFIG)
    return net(data, is_training)

###################################
#########  Transformer  ###########
###################################

def layer_norm(x: jax.Array,  name: Optional[str] = None) -> jax.Array:
  """Applies a unique LayerNorm to x with default settings."""
  ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name = name)
  return ln(x)


"""Didactic exasmple of an autoregressive Transformer-based language model.
Glossary of shapes:
- B: Batch size.
- T: Sequence length.
- D: Model embedding size.
- H: Number of attention heads.
- V: Vocabulary size.
"""
@dataclasses.dataclass
class LanguageModel(hk.Module):
  """An autoregressive transformer-based language model."""

  #transformer: Transformer
  transformer: hk_transformer.UTBlock
  model_size: int
  vocab_size: int
  pad_token: int
  name: Optional[str] = None

  def __call__(
      self,
      tokens: jax.Array,
      *,
      is_training: bool = True,
  ) -> jax.Array:
    """Forward pass, producing a sequence of logits."""
    input_mask = jnp.greater(tokens, self.pad_token)
    unused_batch_size, seq_len = tokens.shape

    # Embed the input tokens and positions.
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(
        self.vocab_size, embed_dim=self.model_size, w_init=embed_init)
    token_embeddings = token_embedding_map(tokens)
    positional_embeddings = hk.get_parameter(
        'positional_embeddings', [seq_len, self.model_size], init=embed_init)
    input_embeddings = token_embeddings + positional_embeddings  # [B, T, D]

    h = jnp.zeros_like(input_embeddings)
    # Run the transformer over the inputs.
    embeddings = self.transformer(
        h,
        input_embeddings,
        input_mask,
        is_training=is_training,
    )  # [B, T, D]


    # Decode the embeddings (here, we use untied weights).
    return hk.Linear(self.vocab_size, name="player_logits")(embeddings), embeddings  # [B, T, V]
  
# Create the model.
def transformer_shakespeare(tokens, num_layers = 6,  num_heads = 8,
                            model_size = 32, vocab_size = 90, pad_token = 0,
                            key_size = 32, dropout_rate = 0.1):
    lm = LanguageModel(model_size=model_size,
                       vocab_size=vocab_size,
                       pad_token=pad_token,
                       transformer= hk_transformer.UTBlock(num_heads=num_heads,
                                               num_layers=num_layers,
                                               #key_size=key_size,
                                               dropout_rate=dropout_rate))
    return lm(tokens)
  

######################################
#########  DEQ-Transformer ###########
######################################
def deq_transformer(vocab_size: int,
                    d_model: int,
                    num_heads: int,
                    num_layers: int,
                    dropout_rate: float,
                    max_iter: int = 20):
    """Create the model's forward pass."""

    def forward_fn(tokens,
                   is_training: bool = True) -> jnp.ndarray:
        """Forward pass."""
        #tokens = data['obs']
        input_mask = jnp.greater(tokens, 0)
        batch_size, seq_length = tokens.shape

        # Embed the input tokens and positions.
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
        input_embeddings = token_embedding_map(tokens)
        positional_embeddings = hk.get_parameter(
            'pos_embs', [seq_length, d_model], init=embed_init)

        x = input_embeddings + positional_embeddings
        h = jnp.zeros_like(x)

        # Create transformer block
        transformer_block = hk_transformer.UTDEQBlock(num_heads=num_heads,
                                                      num_layers=num_layers,
                                                      dropout_rate=dropout_rate)

        
        transformed_net = hk.transform(transformer_block)

        inner_params = hk.experimental.lift(transformed_net.init, 
                                            name="DEQTransformer_Block")(hk.next_rng_key(), 
                                                                         h, x, input_mask, 
                                                                         is_training)

        def f(_params, _rng, _z, *args): 
            return transformed_net.apply(_params, _rng, _z, *args, is_training = is_training)
        z_star = deq(inner_params, hk.next_rng_key(), h, f, max_iter, x, input_mask)

        transformer_last= hk_transformer.UTDEQBlock(num_heads=num_heads,
                                                    num_layers=1,
                                                    dropout_rate=dropout_rate,
                                                    name = "player")
        embeddings = transformer_last(
            z_star,
            input_embeddings,
            input_mask,
            is_training=is_training,
        )  # [B, T, D]
        

        return hk.Linear(vocab_size, name="player_logits")(embeddings), z_star

    return forward_fn