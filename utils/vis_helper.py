from matplotlib import pyplot as plt
import numpy as np
import torch

def visualize_point_clouds_3d_list(pcl_lst, title_lst, vis_order, vis_2D, bound, S, labels):
    t_list = []
    for i in range(len(pcl_lst)):
        img = visualize_point_clouds_3d([pcl_lst[i]], [title_lst[i]] if title_lst is not None else None,
                                        vis_order, vis_2D, bound, S, [labels[i]] if labels is not None else None)
        t_list.append(img)
    img = np.concatenate(t_list, axis=2)
    return img

def visualize_point_clouds_3d(pcl_lst, title_lst=None,
                              vis_order=[2, 0, 1], vis_2D=1, bound=1.5, S=3, labels=None):
    """
    Copied and modified from https://github.com/stevenygd/PointFlow/blob/b7a9216ffcd2af49b24078156924de025c4dbfb6/utils.py#L109 

    Args: 
        pcl_lst: list of tensor, len $L$ = num of point sets, 
            each tensor in shape (N,3), range in [-1,1] 
    Returns: 
        image with $L$ column 
    """
    assert(type(pcl_lst) == list and torch.is_tensor(pcl_lst[0])
           ), f'expect list of tensor, get {type(pcl_lst)} and {type(pcl_lst[0])}'
    if len(pcl_lst) > 1:
        return visualize_point_clouds_3d_list(pcl_lst, title_lst, vis_order, vis_2D, bound, S, labels)

    pcl_lst = [pcl.cpu().detach().numpy() for pcl in pcl_lst]
    labels = [l.cpu().detach().numpy() if torch.is_tensor(l) else l for l in labels]

    if title_lst is None:
        title_lst = [""] * len(pcl_lst)

    if labels is None:
        labels = [None] * len(pcl_lst)

    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    num_col = len(pcl_lst)
    assert(num_col == len(title_lst)
           ), f'require same len, get {num_col} and {len(title_lst)}'
    
    for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
        ax1 = fig.add_subplot(1, num_col, 1 + idx, projection='3d')
        ax1.set_title(title)

        if labels[idx] is None:
            labels = [np.zeros(pts.shape[0], dtype=np.int32)]

        label_arr = labels[idx]

        cmap = plt.get_cmap('Set1')  # More contrasting colors for few classes
        norm = plt.Normalize(vmin=label_arr.min(), vmax=label_arr.max())
        colors = cmap(norm(label_arr))

        if type(S) is list:
            psize = S[idx]
        else:
            psize = S

        ax1.scatter(pts[:, vis_order[0]], pts[:, vis_order[1]],
                    pts[:, vis_order[2]], s=psize, c=colors)
        ax1.set_xlim(-bound, bound)
        ax1.set_ylim(-bound, bound)
        ax1.set_zlim(-bound, bound)
        ax1.grid(False)

    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = fig2data(fig)
    res = np.transpose(res, (2, 0, 1))  # 3,H,W

    plt.close()

    if vis_2D:
        v1 = 0.5
        v2 = 0
        fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
        num_col = len(pcl_lst)
        assert(num_col == len(title_lst)
               ), f'require same len, get {num_col} and {len(title_lst)}'
        for idx, (pts, title) in enumerate(zip(pcl_lst, title_lst)):
            ax1 = fig.add_subplot(1, num_col, 1 + idx, projection='3d')

            assert(len(labels) == len(pcl_lst)), f'require same len, get {len(labels)} and {len(pcl_lst)}'

            label_arr = labels[idx]
            if torch.is_tensor(label_arr):
                label_arr = label_arr.cpu().detach().numpy()
            cmap = plt.get_cmap('Set1')  # More contrasting colors for few classes
            norm = plt.Normalize(vmin=label_arr.min(), vmax=label_arr.max())
            colors = cmap(norm(label_arr))

            if type(S) is list:
                psize = S[idx]
            else:
                psize = S
            ax1.scatter(pts[:, vis_order[0]], pts[:, vis_order[1]],
                        pts[:, vis_order[2]], s=psize, c=colors)
            ax1.set_xlim(-bound, bound)
            ax1.set_ylim(-bound, bound)
            ax1.set_zlim(-bound, bound)
            ax1.grid(False)
            ax1.set_title(title + '-2D')
            ax1.view_init(v1, v2)  # 0.5, 0)

        fig.canvas.draw()

        # grab the pixel buffer and dump it into a numpy array
        # res_2d = np.array(fig.canvas.renderer._renderer)
        res_2d = fig2data(fig)
        res_2d = np.transpose(res_2d, (2, 0, 1))
        plt.close()

        res = np.concatenate([res, res_2d], axis=1)
    return res


def fig2data(fig):
    """
    Adapted from https://stackoverflow.com/questions/55703105/convert-matplotlib-figure-to-numpy-array-of-same-shape 
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    ## fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


if __name__ == "__main__":
    # Example usage
    pcl = [torch.rand(100, 3) * 2 - 1, torch.rand(100, 3) * 2 - 1]  # Random point cloud
    title_lst = ["PC1", "PC2"]
    labels = [torch.randint(0, 4, (100,)), torch.randint(0, 4, (100,))]
    img = visualize_point_clouds_3d(pcl, title_lst, [2, 0, 1], vis_2D=1, bound=1.5, S=3, labels=labels)
    plt.imshow(img.transpose(1, 2, 0))  # Transpose to HWC for displaying
    plt.show()