from helpers import get_dataset_segs
import napari

directory = '/Users/amcg0011/Data/pia-tracking/dl-results/segmentations'
gt, seg, dog, im = get_dataset_segs(directory)

v = napari.view_image(im, scale=(1, 4, 1, 1), name='Images')
v.add_labels(gt, scale=(1, 4, 1, 1), name='Ground Truth')
v.add_labels(seg, scale=(1,4,1,1), name='DL')
v.add_labels(dog, scale=(1,4,1,1), name='DoG')
napari.run()