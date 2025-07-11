from .odefunc import ODEfunc, ODEnet
from .norm import MovingBatchNorm1d
from .cnf import CNF, SequentialFlow

def build_model(cfg, input_dim, hidden_dims, context_dim, num_blocks, conditional):
    def build_cnf():
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(input_dim,),
            context_dim=context_dim,
            layer_type=cfg.layer_type,
            nonlinearity=cfg.nonlinearity,
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=cfg.time_length,
            train_T=cfg.train_T,
            conditional=conditional,
            solver=cfg.solver,
            use_adjoint=cfg.use_adjoint,
            atol=cfg.atol,
            rtol=cfg.rtol,
        )
        return cnf

    chain = [build_cnf() for _ in range(num_blocks)]
    if cfg.batch_norm:
        bn_layers = [MovingBatchNorm1d(input_dim, bn_lag=cfg.bn_lag, sync=cfg.sync_bn)
                     for _ in range(num_blocks)]
        bn_chain = [MovingBatchNorm1d(input_dim, bn_lag=cfg.bn_lag, sync=cfg.sync_bn)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = SequentialFlow(chain)

    return model


def get_point_cnf(cfg):
    dims = tuple(map(int, cfg.dims.split("-")))
    model = build_model(cfg, cfg.input_dim, dims, cfg.latent_dim, cfg.num_blocks, True).cuda()
    return model