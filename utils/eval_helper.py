from .eval_metrics import EMD_CD_F1, compute_all_metrics, jsd_between_point_cloud_sets
from .vis_helper import visualize_point_clouds_3d
from .data_helper import normalize_point_clouds

from loguru import logger
import numpy as np
import os
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
    grid = torchvision.utils.make_grid(img_list, nrow=max(1, num_recon // 2))
    if writer is not None:
        writer.add_image(titles, grid, step)

def compute_NLL_metric(gen_pcs, ref_pcs, device, writer=None, batch_size=200, step=-1, tag=''):
    # evaluate the reconstrution results
    metrics = EMD_CD_F1(gen_pcs.to(device), ref_pcs.to(device),
                     batch_size=batch_size, accelerated_cd=True, reduced=False)
    titles = 'nll/first-10-%s' % tag
    k1, k2, k3 = list(metrics.keys())
    nvis = min(10, gen_pcs.shape[0])
    subtitles = [
        [
            "ori",
            f"gen-{k1}={metrics[k1][j]*1e2:.1f}x1e-2;{k2}={metrics[k2][j]*1e2:.1f}x1e-2;{k3}={metrics[k3][j]*1e2:.1f}x1e-2"
        ]
        for j in range(nvis)
    ]
    pair_vis(gen_pcs[:nvis], ref_pcs[:nvis], titles, subtitles, writer, step=step)
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
def compute_score(gen, ref, batch_size_test=256, device_str='cuda',
                  device=None, accelerated_cd=True, writer=None,
                  visualize=False, norm_vis=False, norm_box=False,
                  mean=None, std=None, save_vis_path=None, step=0):
    """Compute MMD / COV / 1-NNA / JSD between generated and reference clouds.

    Args:
        gen: generated point clouds, either a tensor (Nsample, Npoint, 3 or 6)
            or a path to a .pt file holding one.
        ref: reference point clouds, either a tensor (Nref, Npoint, 3 or 6) or
            a path to a .pt file saved as
            `torch.save({'ref': ref_pcs, 'mean': m_pcs, 'std': s_pcs}, path)`.
        mean, std: per-shape normalisation stats, used only when `ref` is a
            tensor. If both are None the inputs are taken to be already in
            world scale and no denormalisation is applied.
        save_vis_path: where to write the comparison png. Defaults to a path
            next to `gen` when `gen` is a path; when it is None no file is
            written (the image is still sent to the writer).
        step: global step used when logging images to the writer.

    Returns:
        dict of metric name -> float.
    """
    logger.info(
        '[Compute sample metric] sample: {} and ref: {}',
        gen if isinstance(gen, str) else 'tensor%s' % (tuple(gen.shape),),
        ref if isinstance(ref, str) else 'tensor%s' % (tuple(ref.shape),))

    # -- resolve references -- #
    if isinstance(ref, str):
        ref_obj = torch.load(ref, weights_only=False)
        ref_pcs = ref_obj['ref']
        m_pcs, s_pcs = ref_obj['mean'], ref_obj['std']
    else:
        ref_pcs = ref
        m_pcs, s_pcs = mean, std

    # -- resolve samples -- #
    if isinstance(gen, str):
        if save_vis_path is None:
            save_vis_path = gen.replace('.pt', '_eval.png')
        gen_pcs = torch.load(gen, weights_only=False)
    else:
        gen_pcs = gen

    if isinstance(gen_pcs, dict):
        logger.warning('the gen_pcs is a dict with keys {}; usually it is a '
                       'tensor -- you may have loaded the training data',
                       list(gen_pcs.keys()))
        gen_pcs = gen_pcs['ref']

    gen_pcs = torch.as_tensor(gen_pcs).cpu()
    ref_pcs = torch.as_tensor(ref_pcs).cpu()

    # Match the point counts (generation may use more points than the ref).
    if gen_pcs.shape[1] > ref_pcs.shape[1]:
        xperm = np.random.permutation(np.arange(gen_pcs.shape[1]))[
            :ref_pcs.shape[1]]
        gen_pcs = gen_pcs[:, xperm]

    device = torch.device(device_str) if device is None else device

    # Match the shape counts: metrics such as 1-NNA assume equal set sizes.
    N_ref = min(ref_pcs.shape[0], gen_pcs.shape[0])
    if ref_pcs.shape[0] != gen_pcs.shape[0]:
        logger.warning('[Compute sample metric] gen has {} shapes but ref has '
                       '{}; truncating both to {}',
                       gen_pcs.shape[0], ref_pcs.shape[0], N_ref)
    ref_pcs = ref_pcs[:N_ref]
    gen_pcs = gen_pcs[:N_ref]
    if m_pcs is not None:
        m_pcs = torch.as_tensor(m_pcs).cpu()[:N_ref]
    if s_pcs is not None:
        s_pcs = torch.as_tensor(s_pcs).cpu()[:N_ref]

    logger.info('[Data shape] ref_pcs: {}, gen_pcs: {}, mean={}, std={}; norm_box={}',
                tuple(ref_pcs.shape), tuple(gen_pcs.shape),
                None if m_pcs is None else tuple(m_pcs.shape),
                None if s_pcs is None else tuple(s_pcs.shape), norm_box)

    if gen_pcs.shape[2] > 3:  # B,N,3 or 6
        gen_pcs = gen_pcs[:, :, :3]
    if ref_pcs.shape[2] > 3:
        ref_pcs = ref_pcs[:, :, :3]

    if norm_box:
        ref_pcs = 0.5 * torch.stack(
            normalize_point_clouds([p for p in ref_pcs]), dim=0)
        gen_pcs = 0.5 * torch.stack(
            normalize_point_clouds([p for p in gen_pcs]), dim=0)
    elif m_pcs is not None and s_pcs is not None:
        ref_pcs = ref_pcs * s_pcs + m_pcs
        gen_pcs = gen_pcs * s_pcs + m_pcs
    else:
        logger.info('[Compute sample metric] no mean/std supplied; treating '
                    'the inputs as already denormalised')
    # visualize first few samples:
    if visualize:
        img_list = []
        gen_list = []
        ref_list = []
        for i in range(min(20, N_ref)):
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

        grid = torchvision.utils.make_grid(gen_list)
        if writer is not None:
            writer.add_image('gen/samples', grid, step)

        ref_grid = torchvision.utils.make_grid(ref_list)
        if writer is not None:
            writer.add_image('gen/refs', ref_grid, step)
            writer.add_image('gen/samples_vs_ref',
                             torch.cat([grid, ref_grid], dim=1), step)

        if save_vis_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_vis_path)), exist_ok=True)
            torchvision.utils.save_image(img_list, save_vis_path)
            logger.info('Save vis at: {}', save_vis_path)

    results = compute_all_metrics(gen_pcs.to(device).float(),
                                  ref_pcs.to(device).float(), batch_size_test, accelerated_cd=accelerated_cd)

    jsd = jsd_between_point_cloud_sets(
        gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    results['jsd'] = jsd

    scalars, skipped = {}, []
    for k, v in results.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                scalars[k] = v.item()
            else:
                skipped.append(k)
        elif isinstance(v, (int, float, np.floating, np.integer)):
            scalars[k] = float(v)
        else:
            skipped.append(k)
    if skipped:
        logger.debug('[Compute sample metric] dropped non-scalar entries: {}',
                     skipped)

    return scalars