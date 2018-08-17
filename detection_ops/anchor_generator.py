import numpy as np


def _tile_anchors(grid_height,
                 grid_width,
                 scale,
                 aspect_ratio,
                 anchor_stride,
                 anchor_offset,
                 base_anchor_size=None):
  """Create a tiled set of anchors strided along a grid in image space.

  This op creates a set of anchor boxes by placing a "basis" collection of
  boxes with user-specified scales and aspect ratios centered at evenly
  distributed points along a grid.  The basis collection is specified via the
  scale and aspect_ratio arguments.  For example, setting scales=[.1, .2, .2]
  and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
  .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
  and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
  placing it over its respective center.

  Grid points are specified via grid_height, grid_width parameters as well as
  the anchor_stride and anchor_offset parameters.

  Args:
    grid_height: size of the grid in the y direction (int or int scalar tensor)
    grid_width: size of the grid in the x direction (int or int scalar tensor)
    scale: a 1-d  (float) tensor representing the scale of each box in the
      basis set.
    aspect_ratio: a 1-d (float) tensor representing the aspect ratio of each
      box in the basis set.  The length of the scale and aspect_ratio tensors
      must be equal.
    anchor_stride: difference in centers between base anchors for adjacent grid
                   positions (float tensor of shape [2])
    anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                   upper left element of the grid, this should be zero for
                   feature networks with only VALID padding and even receptive
                   field size, but may need some additional calculation if other
                   padding is used (float tensor of shape [2])
    base_anchor_size: base anchor size as [height, width]
      (float tensor of shape [2])
  Returns:
    a BoxList holding a collection of N anchor boxes
  """
  if not len(scale) == len(aspect_ratio):
    raise ValueError('scale must be a list with the same '
                     'length aspect_ratio')
  if not base_anchor_size:
    base_anchor_size = anchor_stride
  ratio_sqrt = np.sqrt(aspect_ratio)
  heights = scale / ratio_sqrt * base_anchor_size[0]
  widths = scale * ratio_sqrt * base_anchor_size[1]

  # Get a grid of box centers
  y_centers = np.arange(grid_height)
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = np.arange(grid_width)
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = np.meshgrid(x_centers, y_centers)

  widths_grid, x_centers_grid = np.meshgrid(widths, x_centers)
  heights_grid, y_centers_grid = np.meshgrid(heights, y_centers)
  bbox_centers = np.stack([y_centers_grid, x_centers_grid], axis=2)
  bbox_sizes = np.stack([heights_grid, widths_grid], axis=2)
  bbox_centers = np.reshape(bbox_centers, [-1, 2])
  bbox_sizes = np.reshape(bbox_sizes, [-1, 2])
  bbox_corners = np.concatenate((bbox_centers - .5 * bbox_sizes, bbox_centers + .5 * bbox_sizes), 1)
  return bbox_corners


def generate_anchors(feature_map_dims, scales, aspect_ratios):
  """Generate anchors based on the size of feature maps"""
  if not (isinstance(scales, list)
          and all([isinstance(list_item, list) for list_item in scales])):
    raise ValueError('scales must be a list of lists.')
  if not (isinstance(aspect_ratios, list)
          and all([isinstance(list_item, list) for list_item in aspect_ratios])):
    raise ValueError('aspect_ratios must be a list of lists.')
  if not (isinstance(feature_map_dims, list)
          and len(feature_map_dims) == len(scales) == len(aspect_ratios)):
    raise ValueError('feature_map_dims must be a list with the same '
                     'length as scales and aspect_ratios')
  if not all([isinstance(list_item, tuple) and len(list_item) == 2
              for list_item in feature_map_dims]):
    raise ValueError('feature_map_dims must be a list of pairs.')

  anchor_strides = [(1.0 / np.float(pair[0]), 1.0 / np.float(pair[1]))
                    for pair in feature_map_dims]
  anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
                    for stride in anchor_strides]

  anchor_grid_list = []
  for grid_size, scale, aspect_ratio, stride, offset in zip(
          feature_map_dims, scales, 
          aspect_ratios, anchor_strides, 
          anchor_offsets):
    tiled_anchors = _tile_anchors(grid_height=grid_size[0],
                                 grid_width=grid_size[1],
                                 scale=scale,
                                 aspect_ratio=aspect_ratio,
                                 anchor_stride=stride,
                                 anchor_offset=offset)
    anchor_grid_list.append(tiled_anchors)
  anchor_grid_list = np.concatenate(anchor_grid_list, 0)
  return anchor_grid_list


if __name__ =='__main__':
  anchors = generate_anchors([(7,7),(4,4)], scales=[[0.5,0.8], [0.5,0.8]], aspect_ratios=[[1,1],[1,1]])
  print(anchors)