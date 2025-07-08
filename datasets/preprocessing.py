import numpy as np
from loguru import logger
import h5py as h5
import argparse
import multiprocessing
import os
import gc
import trimesh

# taken from https://github.com/optas/latent_3d_points/blob/
# 8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane',
    '02773838': 'bag',
    '02801938': 'basket',
    '02808440': 'bathtub',
    '02818832': 'bed',
    '02828884': 'bench',
    '02876657': 'bottle',
    '02880940': 'bowl',
    '02924116': 'bus',
    '02933112': 'cabinet',
    '02747177': 'can',
    '02942699': 'camera',
    '02954340': 'cap',
    '02958343': 'car',
    '03001627': 'chair',
    '03046257': 'clock',
    '03207941': 'dishwasher',
    '03211117': 'monitor',
    '04379243': 'table',
    '04401088': 'telephone',
    '02946921': 'tin_can',
    '04460130': 'tower',
    '04468005': 'train',
    '03085013': 'keyboard',
    '03261776': 'earphone',
    '03325088': 'faucet',
    '03337140': 'file',
    '03467517': 'guitar',
    '03513137': 'helmet',
    '03593526': 'jar',
    '03624134': 'knife',
    '03636649': 'lamp',
    '03642806': 'laptop',
    '03691459': 'speaker',
    '03710193': 'mailbox',
    '03759954': 'microphone',
    '03761084': 'microwave',
    '03790512': 'motorcycle',
    '03797390': 'mug',
    '03928116': 'piano',
    '03938244': 'pillow',
    '03948459': 'pistol',
    '03991062': 'pot',
    '04004475': 'printer',
    '04074963': 'remote_control',
    '04090263': 'rifle',
    '04099429': 'rocket',
    '04225987': 'skateboard',
    '04256520': 'sofa',
    '04330267': 'stove',
    '04530566': 'vessel',
    '04554684': 'washer',
    '02992529': 'cellphone',
    '02843684': 'birdhouse',
    '02871439': 'bookshelf',
    '04574864': 'pasta15k',
    '04574865': 'pasta',
    '02858304': 'boat',
    '02834778': 'bicycle'
}

# Reverse mapping for category names to synset IDs
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

def process_obj_file(sample):    
    # Load and clean mesh
    mesh = trimesh.load(sample, process=True, force='mesh')  # process=True does basic cleanup
    
    # Additional cleaning to match your original requirements
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    
    shape_center = mesh.centroid

    # Center the mesh
    mesh.apply_translation(-shape_center)
    
    # Scale to unit sphere
    shape_scale = np.sqrt((mesh.vertices**2).sum(1)).max()
    mesh.apply_scale(1.0 / shape_scale)

    mins, maxs = mesh.bounds
    bbox_c = (maxs + mins) / 2.
    bbox_s = (maxs - mins).max()

    return {
        'vertices_c': mesh.vertices.astype(np.float32),
        'orig_c': shape_center.astype(np.float32),
        'orig_s': np.float32(shape_scale),
        'bbox_c': bbox_c.astype(np.float32),
        'bbox_s': np.float32(bbox_s),
        'faces_vc': mesh.faces.astype(np.uint32)
    }
        

def process_meshes_memory_efficient(sample_paths, n_workers, progress_key="meshes"):
    """
    Processes all meshes in parallel and aggregates results efficiently.
    """
    logger.info(f"Processing {len(sample_paths)} {progress_key} meshes with {n_workers} workers")
    
    # Process all meshes in parallel using a single pool
    with multiprocessing.Pool(processes=n_workers) as pool:
        processing_results = pool.map(process_obj_file, sample_paths)
    
    # Pre-allocate arrays based on total counts
    total_vertices = sum(len(result['vertices_c']) for result in processing_results)
    total_faces = sum(len(result['faces_vc']) for result in processing_results)
    n_meshes = len(processing_results)
    
    logger.info(f"Aggregating {total_vertices} vertices and {total_faces} faces from {n_meshes} meshes")
    
    # Pre-allocate output arrays
    all_vertices = np.empty((total_vertices, 3), dtype=np.float32)
    all_faces = np.empty((total_faces, 3), dtype=np.uint32)
    
    # Bounds arrays
    vertices_bounds = np.zeros(n_meshes + 1, dtype=np.uint64)
    faces_bounds = np.zeros(n_meshes + 1, dtype=np.uint64)
    
    # Metadata arrays
    orig_c = np.empty((n_meshes, 3), dtype=np.float32)
    orig_s = np.empty(n_meshes, dtype=np.float32)
    bbox_c = np.empty((n_meshes, 3), dtype=np.float32)
    bbox_s = np.empty(n_meshes, dtype=np.float32)
    
    # Fill arrays efficiently
    v_offset = 0
    f_offset = 0
    
    for i, result in enumerate(processing_results):
        # Vertices
        n_verts = len(result['vertices_c'])
        if n_verts > 0:
            all_vertices[v_offset:v_offset + n_verts] = result['vertices_c']
        vertices_bounds[i + 1] = vertices_bounds[i] + n_verts
        v_offset += n_verts
        
        # Faces
        n_faces = len(result['faces_vc'])
        if n_faces > 0:
            all_faces[f_offset:f_offset + n_faces] = result['faces_vc']
        faces_bounds[i + 1] = faces_bounds[i] + n_faces
        f_offset += n_faces
        
        # Metadata
        orig_c[i] = result['orig_c']
        orig_s[i] = result['orig_s']
        bbox_c[i] = result['bbox_c']
        bbox_s[i] = result['bbox_s']
    
    # Clean up processing results to free memory
    del processing_results
    gc.collect()
    
    logger.info(f"Successfully processed all {progress_key} meshes")
    
    return {
        'all_vertices': all_vertices,
        'all_faces': all_faces,
        'vertices_bounds': vertices_bounds,
        'faces_bounds': faces_bounds,
        'orig_c': orig_c,
        'orig_s': orig_s,
        'bbox_c': bbox_c,
        'bbox_s': bbox_s
    }

def preprocessing(args):
    assert os.path.exists(args.data_dir), f"Data path {args.data_dir} does not exist"
    assert isinstance(args.categories, list), "Categories should be a list"

    n_workers = args.n_processes
    train_split = args.train_split
    test_split = args.test_split

    assert train_split + test_split <= 1, "Train and test splits must sum to less than 1"

    val_split = 1 - train_split - test_split

    folder_list = []
    out_list = []

    if "all" in args.categories:
        categories = os.listdir(args.data_dir)
        folder_list = [os.path.join(args.data_dir, category) for category in categories if os.path.isdir(os.path.join(args.data_dir, category))]
        out_list = [os.path.join(args.save_dir, category) for category in categories if os.path.isdir(os.path.join(args.data_dir, category))]
    elif args.dataset == "ShapeNetCore.v2":
        synset_ids = [cate_to_synsetid[c] for c in args.categories]
        for synset_id, category in zip(synset_ids, args.categories):
            assert os.path.exists(os.path.join(args.data_dir, synset_id)), f"Category {category} does not exist in {args.data_dir}"
            folder_list.append(os.path.join(args.data_dir, synset_id))
            out_list.append(os.path.join(args.save_dir, synset_id))
    else:
        for category in args.categories:
            assert os.path.exists(os.path.join(args.data_dir, category)), f"Category {category} does not exist in {args.data_dir}"
            folder_list.append(os.path.join(args.data_dir, category))
            out_list.append(os.path.join(args.save_dir, category))

    logger.info(f"[DATA] Preprocessing categories: {args.categories}, data path: {args.save_dir}")

    samples = {
        'train': [],
        'val': [],
        'test': []
    }
    cat_samples = []

    for cat_path, out_path in zip(folder_list, out_list):
        logger.info(f"Processing category: {cat_path}")
        for shape_path in os.listdir(cat_path):
            if os.path.exists(os.path.join(cat_path, shape_path, 'models', 'model_normalized.obj')):
                obj_path = os.path.join(cat_path, shape_path, 'models', 'model_normalized.obj')
                cat_samples.append(obj_path)
            else:
                logger.warning(f"Model normalized file does not exist for {shape_path}, skipping this shape.")

        train_samples, val_samples, test_samples = np.split(cat_samples, [int(train_split * len(cat_samples)), int((train_split + val_split) * len(cat_samples))])

        # samples['train'].extend(train_samples)
        # samples['val'].extend(val_samples)
        # samples['test'].extend(test_samples)

        samples['train'] = train_samples
        samples['val'] = val_samples
        samples['test'] = test_samples

        os.makedirs(out_path, exist_ok=True)
        logger.info(f"Saving processed samples to {out_path}")

        with h5.File(os.path.join(out_path, 'dataset.h5'), 'w') as f:
            for key, sample in samples.items():
                logger.info(f"Processing {len(sample)} samples for {key} set.")

                group = f.create_group(key)

                # Create datasets #
                vcb_ds = group.create_dataset('vertices_c_bounds', shape=(len(sample) + 1,), dtype=np.uint64)
                vcb_ds[0] = 0
                vc_ds = group.create_dataset('vertices_c', shape=(0, 3), maxshape=(None, 3), dtype=np.float32)

                orig_c_ds = group.create_dataset('orig_c', shape=(len(sample), 3), dtype=np.float32)
                orig_s_ds = group.create_dataset('orig_s', shape=(len(sample),), dtype=np.float32)

                bbox_c_ds = group.create_dataset('bbox_c', shape=(len(sample), 3), dtype=np.float32)
                bbox_s_ds = group.create_dataset('bbox_s', shape=(len(sample),), dtype=np.float32)

                fb_ds = group.create_dataset('faces_bounds', shape=(len(sample) + 1,), dtype=np.uint64)
                fb_ds[0] = 0
                fvc_ds = group.create_dataset('faces_vc', shape=(0, 3), maxshape=(None, 3), dtype=np.uint32)

                results = process_meshes_memory_efficient(sample, n_workers, progress_key=key)
                
                # Write to HDF5 datasets
                vc_ds.resize((len(results['all_vertices']), 3))
                vc_ds[:] = results['all_vertices']
                
                fvc_ds.resize((len(results['all_faces']), 3))
                fvc_ds[:] = results['all_faces']
                
                vcb_ds[:] = results['vertices_bounds']
                fb_ds[:] = results['faces_bounds']
                
                orig_c_ds[:] = results['orig_c']
                orig_s_ds[:] = results['orig_s']
                bbox_c_ds[:] = results['bbox_c']
                bbox_s_ds[:] = results['bbox_s']
                
                logger.info(f"Successfully written {len(sample)} meshes to HDF5")

def get_args():
    parser = argparse.ArgumentParser(
        description='Data processor for ShapeNetCore dataset. '
        'All OBJ files are preprocessed and accumulated in a single .h5 file.'
    )
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory containing the unpacked dataset.')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to directory for the output.')
    parser.add_argument('--dataset', type=str, default='ShapeNetCore.v2', choices=['ShapeNetCore.v2', 'custom'], help='Dataset to preprocess.')
    parser.add_argument('--categories', default=['all'], nargs='+', help='List of categories to preprocess. Use "all" for all categories.')
    parser.add_argument('--n_processes', type=int, help='Number of parallel processing jobs.')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of data for training.')
    parser.add_argument('--test_split', type=float, default=0.1, help='Proportion of data for testing.')

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    logger.info(f"Arguments: {args}")

    preprocessing(args)

if __name__ == '__main__':
    main()


