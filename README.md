# KittiPy
**Python functions to import and display the images and labels from the [Kitty Object Detection Evaluation 2012 Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).**

![Examples of uses of KittiPy](https://i.imgur.com/STCsquT.jpg)

## Prerequisites

- [python](https://www.python.org/) ≥ _v2.7.6_
- [scipy](https://www.scipy.org/) ≥ _v0.19.1_                  
- [matplotlib](https://matplotlib.org/) ≥ _v1.5.2_
- [numpy](http://www.numpy.org/) ≥ _v1.13.3_

## Installing

1. Copy the file `kittipy.py` in the same directory as the python script or Jupyter notebook you want to use it with.
2. _(Optional)_ Modify the `kittipy.py` global parameters at the begining
3. At the begining of your file, add the line: `from kittipy import *`
4. Enjoy!

## How to use it
### Explore Folder
To get the **numpy array** containing all the ids of the images contained in a folder, use the function:
```
get_data_list(im_set, db_absolute_path = ABSOLUTE_PATH)
```
| Arguments             | Default                   | Description                                    |
| ----------------------|---------------------------|------------------------------------------------|
| `im_set`              |                           | **[string]** 'train' or 'test' to chose the type of set  |
| `db_absolute_path`    | _ABSOLUTE_PATH_           | **[string]** absolute path to the Kitti root folder |



### Import Image
To import images as a **numpy array**, use the function:
```
import_im(im_id, im_set, db_absolute_path = ABSOLUTE_PATH)
```
| Arguments             | Default                   | Description                                    |
| ----------------------|---------------------------|------------------------------------------------|
| `im_id`               |                           | **[int]** corresponding to the image id in the kitti dataset  |
| `im_set`              |                           | **[string]** 'train' or 'test' to chose the type of set  |
| `db_absolute_path`    | _ABSOLUTE_PATH_           | **[string]** absolute path to the Kitti root folder |

### Import Labels
To import the labels as a **list of python dictionaries**, use the function:

```
import_labels(im_id, im_set, db_absolute_path = ABSOLUTE_PATH)
```
| Arguments             | Default                   | Description                                    |
| ----------------------|---------------------------|------------------------------------------------|
| `im_id`               |                           | **[int]** corresponding to the image id in the kitti dataset  |
| `im_set`              |                           | **[string]** 'train' or 'test' to chose the type of set  |
| `db_absolute_path`    | _ABSOLUTE_PATH_           | **[string]** absolute path to the Kitti root folder |

#### Description of the fields of the dictionary
| Fields                     | Description                                    |
| ------------------------------|------------------------------------------------|
| `type`                        | **[string]** 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'  |
| `truncated`                   | **[float]** Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving the image boundaries  |
| `occluded`                    | **[int]** Integer (0,1,2,3) indicating occlusion state: 0 = fully visible, 1 = partly occluded, 2 = largely occluded, 3 = unknown  |
| `alpha`                       | **[float]** Observation angle of an object [-pi..pi]   |
| `bbox`                        | **[dict][float]** `x_min`, `y_min`, `x_max`, `y_max` pixel coordinates (0-based index)  |
| `3D_dim`                      | **[dict][float]** `height`, `width`, `length` (in meters)   |
| `3D_loc`                      | **[dict][float]** `x`, `y`, `z` in camera coordinates (in meters)  |
| `rotation_y`                  | **[float]** around Y-axis in camera coordinates [-pi..pi] |
| `score` _(Optional)_          | **[float]** Only for results, indicating confidence in detection, needed for p/r curves, higher is better.  |


### Display Image + Labels
To display a image and its labels in a **Jupyter Notebook**, use the function:
```
display_im(im, labels = [], display_boxes = True, display_info = True, 
           types_to_display = DEFAULT_TYPES_TO_DISPLAY, 
           info_to_display = DEFAULT_INFO_TO_DISPLAY, 
           db_absolute_path = ABSOLUTE_PATH, im_width = FIG_WIDTH, 
           im_height = FIG_HEIGHT, display_axis = False, 
           title = '', display_center_boxes = True, num_cell_grid = 0)
```
| Arguments             | Default                   | Description                |
| ----------------------|---------------------------|----------------------------|
| `im`                  |                           | **[numpy.array]** Image to display  |
| `labels`              | _[]_                      | **[list][dict]** List of dictionaries containing the labels of the image  |
| `display_boxes`       | _True_                    | **[bool]** Display the boxes around the objects |
| `display_info`        | _True_                    | **[bool]** Write the `info_to_display` under the image  |
| `types_to_display`    | _DEFAULT_TYPES_TO_DISPLAY_| **[list]** List of types of object to consider |
| `info_to_display`     | _DEFAULT_INFO_TO_DISPLAY_ | **[list]** List of characteritics of object to display |
| `im_width`            | _FIG_WIDTH_               | **[int]** Width of the image to display |
| `im_height`           | _FIG_HEIGHT_              | **[int]** Height of the image to display |
| `display_axis`        | _False_                   | **[bool]** Display the axis of the image |
| `title`               | _''_                      | **[string]** Title to write before the image |
| `display_center_boxes`| _True_                    | **[bool]** Display the center of the boxes |
| `num_cell_grid`       | _0_                       | **[int]** Display a `num_cell_grid` x `num_cell_grid` over the image|

**Remarks**: To change `FIG_FONT_SIZE_TITLE`, `COLOR_GRID` and `COLOR_TYPE` do it in the global variables at the begining of the `kittipy.py` file.

### Example
**Code** _(Jupyter Notebook)_
```python
list_ids = get_data_list('train')
image = import_im(list_ids[3], 'train')
labels = import_labels(list_ids[3], 'train')
display_im(image, labels, num_cell_grid = 10)
```
**Output**

![Output of the code above](https://i.imgur.com/tOG55Qs.png)

## Author

* **Lucas Vandroux (冯凯)** [【Github】](https://github.com/LucasVandroux) [【LinkedIn】](https://www.linkedin.com/in/lucasvandroux/)

I started this project to help me during my Master Thesis at Tsinghua University in China. If you find it useful, don't hesitate to use it and improve it.

---
![Logos Footer](https://i.imgur.com/bCStMxt.png)
