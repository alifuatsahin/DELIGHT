from .eval_metrics import EMD_CD_F1
from .vis_helper import visualize_point_clouds_3d
from .data_helper import normalize_point_clouds

from math import log, pi
from loguru import logger
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

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2