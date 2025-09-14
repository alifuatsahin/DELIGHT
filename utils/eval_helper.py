from .eval_metrics import EMD_CD_F1, compute_all_metrics, jsd_between_point_cloud_sets
from .vis_helper import visualize_point_clouds_3d
from .data_helper import normalize_point_clouds

from loguru import logger
import numpy as np
import torch
import torchvision

def pair_vis(gen_x, tr_x, titles, subtitles, writer, step=-1):
    img_list = []
    num_recon = len(gen_x)
    for i in range(num_recon):
        points = gen_x[i]
        points = normalize_point_clouds([tr_x[i], points])
        img = visualize_point_clouds_3d(points, subtitles[i])
        img_list.append(torch.as_tensor(img) / 255.0)
    grid = torchvision.utils.make_grid(img_list, nrow=num_recon//2)
    if writer is not None:
        writer.add_image(titles, grid, step)

def compute_NLL_metric(gen_pcs, ref_pcs, device, writer=None, batch_size=200, step=-1, tag=''):
    # evaluate the reconstrution results
    metrics = EMD_CD_F1(gen_pcs.to(device), ref_pcs.to(device),
                     batch_size=batch_size, accelerated_cd=True, reduced=False)
    titles = 'nll/first-10-%s' % tag
    k1, k2, k3 = list(metrics.keys())
    subtitles = [
        [
            "ori",
            f"gen-{k1}={metrics[k1][j]*1e2:.1f}x1e-2;{k2}={metrics[k2][j]*1e2:.1f}x1e-2;{k3}={metrics[k3][j]*1e2:.1f}x1e-2"
        ]
        for j in range(10)
    ]
    pair_vis(gen_pcs[:10], ref_pcs[:10], titles, subtitles, writer, step=step)
    results = {}

    for k in metrics.keys():
        sorted, indices = torch.sort(metrics[k])
        worse_ten, worse_score = indices[-10:].cpu(), sorted[-10:].cpu()
        titles = f"nll/worst-{k}-{tag}"
        subtitles = [["ori", f"gen-{k}={worse_score[j]*1e2:.2f}x1e-2"] for j in range(len(worse_score))]
        pair_vis(gen_pcs[worse_ten], ref_pcs[worse_ten],
                 titles, subtitles, writer, step=step)

        metrics[k] = metrics[k].mean()

    results.update({k: v.item() for k, v in metrics.items()})

    output = ''
    for k, v in results.items():
        output += f"{k}={v*1e2:.3f}x1e-2"
        logger.info('{}: {}', k, v)

    return results

def get_ref_num(cats, luo_split=False):
    num_test = {
        'animal': 100,
        'airplane': 405,
        'airplane_ps': 405,
        'chair': 662,
        'chair_ps': 662,
        'car': 352,
        'car_ps': 352,
        'all': 1000,
        'mug': 22,
        'bottle': 43
    }
    if luo_split:
        num_test = {
            'airplane': 607,
            'chair': 989,
            'car': 528
        }

    assert(cats in num_test), f'not found: {cats} in {num_test}'
    return num_test[cats]

@torch.no_grad()
def compute_score(output_name, ref_name, batch_size_test=256, device_str='cuda',
                  device=None, accelerated_cd=True, writer=None,
                  visualize=False, norm_vis=False, norm_box=False):
    """
    Args: 
        output_name (str) path to sample obj: tensor: (Nsample.Npoint.3or6)
        ref_name (str) path to torch obj: 
            torch.save({'ref': ref_pcs, 'mean': m_pcs, 'std': s_pcs}, ref_name)
        print_kwargs (dict): entries: dataset, hash, step, epoch; 
    """
    logger.info('[Compute sample metric] sample: {} and ref: {}',
                output_name, ref_name)
    ref = torch.load(ref_name)
    ref_pcs = ref['ref'][:, :, :3]
    m_pcs, s_pcs = ref['mean'], ref['std']
    gen_pcs = torch.load(output_name)
    if gen_pcs.shape[1] > ref_pcs.shape[1]:
        xperm = np.random.permutation(np.arange(gen_pcs.shape[1]))[
            :ref_pcs.shape[1]]
        gen_pcs = gen_pcs[:, xperm]
    if type(gen_pcs) is dict:
        logger.info('WARNING: the gen_pcs is a dict, with key '
                    'as {}| usuaglly its a tensor '
                    'you perhaps takes the train data,',
                    gen_pcs.keys())
        gen_pcs = gen_pcs['ref']
    device = torch.device(device_str) if device is None else device
    # batch_size_test = ref_pcs.shape[0]
    logger.info('[Data shape] ref_pcs: {}, gen_pcs: {}, mean={}, std={}; norm_box={}',
                ref_pcs.shape, gen_pcs.shape, m_pcs.shape, s_pcs.shape, norm_box)
    N_ref = ref_pcs.shape[0]  # subset it
    m_pcs = m_pcs[:N_ref]
    s_pcs = s_pcs[:N_ref]
    ref_pcs = ref_pcs[:N_ref]
    gen_pcs = gen_pcs[:N_ref]
    if gen_pcs.shape[2] == 6:  # B,N,3 or 6
        gen_pcs = gen_pcs[:, :, :3]
        ref_pcs = ref_pcs[:, :, :3]
    if norm_box:
        ref_pcs = 0.5 * torch.stack(normalize_point_clouds(ref_pcs), dim=0)
        gen_pcs = 0.5 * torch.stack(normalize_point_clouds(gen_pcs), dim=0)
    else:
        ref_pcs = ref_pcs * s_pcs + m_pcs
        gen_pcs = gen_pcs * s_pcs + m_pcs
    # visualize first few samples:
    if visualize:
        img_list = []
        gen_list = []
        ref_list = []
        for i in range(20):
            if norm_vis:
                norm_ref, norm_gen = normalize_point_clouds([
                    ref_pcs[i], gen_pcs[i]])
            else:
                norm_ref = ref_pcs[i]
                norm_gen = gen_pcs[i]
            ref_img = visualize_point_clouds_3d([norm_ref],
                                                [f'ref-{i}'], bound=1.0)  # 0.8)
            gen_img = visualize_point_clouds_3d([norm_gen],
                                                [f'gen-{i}'], bound=1.0)  # 0.8)
            ref_list.append(torch.as_tensor(ref_img) / 255.0)
            gen_list.append(torch.as_tensor(gen_img) / 255.0)
            img_list.append(ref_list[-1])
            img_list.append(gen_list[-1])

        path = output_name.replace('.pt', '_eval.png')

        grid = torchvision.utils.make_grid(gen_list)
        # to 3,H,W to H,W,3
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy()
        if writer is not None:
            writer.add_image(ndarr, 'gen/samples')

        ref_grid = torchvision.utils.make_grid(ref_list)
        # to 3,H,W to H,W,3
        ref_ndarr = ref_grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy()
        ndarr = np.concatenate([ndarr, ref_ndarr], axis=0)
        if writer is not None:
            writer.add_image(ndarr, 'gen/samples_vs_ref')

        torchvision.utils.save_image(img_list, path)
        logger.info('Save vis at: {}', path)

    results = compute_all_metrics(gen_pcs.to(device).float(),
                                  ref_pcs.to(device).float(), batch_size_test, accelerated_cd=accelerated_cd)

    jsd = jsd_between_point_cloud_sets(
        gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

    return results