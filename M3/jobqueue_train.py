from dask_jobqueue import SLURMCluster as Cluster
from dask import delayed
from dask.distributed import Client, as_completed
from train import train_unet, test_unet



suffix = 'first-attempt'
out_dir = '~/rl54/segmentation/unet_training'


def training_wrapped(
                     data, 
                     out_dir=out_dir, 
                     suffix=suffix
                     ):
    """
    Wrapper for unet training function

    Parameters
    ----------
    data: tuple of list of str
        in the form (image_paths, label_paths)
    out_dir: str
        directory to which to save training checkpoints and train data
    suffix:
    """
    unet = train_unet(data[0], data[1], out_dir, suffix)
    unet = test_unet(unet, data[0], data[1], out_dir)
    return unet

# Where does the data live
image_paths = ['~/rl54/segmentation/training_data/cang_training/191113_IVMTR26_Inj3_cang_exp3_t58.zarr', 
               '~/rl54/segmentation/training_data/cang_training/191113_IVMTR26_Inj3_cang_exp3_t74.zarr']
labels_paths = ['~/rl54/segmentation/training_data/cang_training/191113_IVMTR26_Inj3_cang_exp3_labels_58_GT.zarr', 
                '~/rl54/segmentation/training_data/cang_training/191113_IVMTR26_Inj3_cang_exp3_labels_t74_GT.zarr']
data = (image_paths, labels_paths)
cluster = Cluster()
client = Client(cluster)

# feed in the data
queue = as_completed(client.map(training_wrapped, data))
for future in queue:
    i, v = future.result()
    future.cancel()
