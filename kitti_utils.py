# *** IMPORT ***
from scipy import misc                # To import the pictures
import math                           # Various mathematical fcts.
import matplotlib.pyplot as plt       # To display the images
import matplotlib.patches as patches  # To draw on the images
from os import listdir                # To list all what a directory contains
from os.path import isfile, join      # To only select files in a directory

# *** PARAMETERS ***
ABSOLUTE_PATH = "/data2/Kitti/left_12g/"

TRAIN_PATH_IM = "data_object_image_2/training/image_2/"
TEST_PATH_IM = "data_object_image_2/testing/image_2/"

TRAIN_PATH_LABEL = "data_object_label_2/training/label_2/"

FIG_WIDTH = 30
FIG_HEIGHT = 15
FIG_FONT_SIZE_TITLE = 35

DEFAULT_TYPES_TO_DISPLAY = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
DEFAULT_INFO_TO_DISPLAY = ['bbox']
COLOR_TYPE ={
    'Car':            '#ff0000',   # Red
    'Van':            '#ffff00',   # Yellow
    'Truck':          '#ff00ff',   # Fuchsia
    'Pedestrian':     '#33cc33',   # Green
    'Person_sitting': '#00ffff',   # Light Blue
    'Cyclist':        '#ff9933',   # Orange
    'Tram':           '#0000ff',   # Blue      
    'Misc':           '#996633',   # Brown
    'DontCare':       '#9900ff',   # Violet
}

# *** LABELS ***
def create_label_path(im_id, im_set, db_absolute_path = ABSOLUTE_PATH):
    """
    This function create the absolute path to the label of an image
    
    Argument:
    im_id            -- int corresponding to the image id in the kitti dataset
    im_set           -- 'train' or 'test'
    db_absolute_path -- absolute path to the Kitti root folder
    
    Returns:
    label_path -- absolute path to the label
    """
    len_filename = 6
    
    label_filename = str(im_id).zfill(len_filename) + ".txt"
    
    label_path = db_absolute_path + globals()[(im_set +'_path_label').upper()] + label_filename
    
    return label_path

def import_labels(im_id, im_set = 'train', db_absolute_path = ABSOLUTE_PATH):
    """
    This import the labels of an image in a list of dictionnary
    
    Argument:
    im_id            -- int corresponding to the image id in the kitti dataset
    im_set           -- 'train' or 'test'
    db_absolute_path -- absolute path to the Kitti root folder
    
    Returns:
    objects -- list of dictionnary containing all the information on the objects in the file
    """
    # Create file absolute path
    label_path = create_label_path(im_id, im_set, db_absolute_path)
    
    objects = []
    
    # Import 
    with open(label_path) as f:
        for line in f:
            # Parse line
            line_parsed = line.strip().split(' ')
            
            # Fill the dictionary with the values
            object_dict = {}
            object_dict['type']        = line_parsed[0]
            object_dict['truncated']   = float(line_parsed[1])
            object_dict['occluded']    = int(line_parsed[2])
            object_dict['alpha']       = float(line_parsed[3])
            object_dict['bbox']        = {'x_min': float(line_parsed[4]),
                                          'y_min': float(line_parsed[5]),
                                          'x_max': float(line_parsed[6]),
                                          'y_max': float(line_parsed[7])}
            object_dict['3D_dim']      = {'height': float(line_parsed[8]),
                                          'width' : float(line_parsed[9]),
                                          'length': float(line_parsed[10])}
            object_dict['3D_loc']      = {'x': float(line_parsed[11]),
                                          'y': float(line_parsed[12]),
                                          'z': float(line_parsed[13])}
            object_dict['rotation_y']  = float(line_parsed[14])
            
            # Add Score [optional in the file]
            if len(line_parsed) > 15:
                object_dict['score']       = float(line_parsed[15])
            
            # Append the dictionary to the list of object in the picture
            objects.append(object_dict)
            
    return objects

def print_labels(labels, type_to_display = DEFAULT_TYPES_TO_DISPLAY, info_to_display = DEFAULT_INFO_TO_DISPLAY):
    """
    This display the information about the objects in a friendly way
    
    Argument:
    labels            -- list of dictionaries containing the information about the object to display
    type_to_display   -- list of types of object to display
    info_to_display   -- list of the name of the information to display from the dictionaries
    
    Returns:
    Print a table with all the information in ASCII
    """
    
    # Variable to control the size of the cells in the table
    len_cell_type = 16
    len_cell = 12

    # Create Header
    header_to_print = '|' + 'type'.ljust(len_cell_type) + '|'
    for info_descr in info_to_display:
        info = labels[0][info_descr]
        if not isinstance(info, dict):
            header_to_print += info_descr.ljust(len_cell - 2).rjust(len_cell) + '|'
        else:
            for key in info:
                header_to_print += key.ljust(len_cell - 2).rjust(len_cell) + '|'
    
    # Print Header
    print(''.ljust(len(header_to_print), '-'))
    print(header_to_print)     
    print(''.ljust(len(header_to_print), '-'))
    print(''.ljust(len(header_to_print), '-'))
    
    # Extract only object corresponding to a certain type
    for type in type_to_display:
        list_dic = [dic for dic in labels if dic['type'] == type]
        
        # Create a str with the information for each object in the list
        for dic in list_dic:
            info_to_print = '|' + dic['type'].ljust(len_cell_type) + '|' 
            for info_descr in info_to_display:
                info = dic[info_descr]
                if not isinstance(info, dict):
                    info_to_print += str(info).ljust(len_cell - 2).rjust(len_cell) + '|'
                else:
                    for _, item in info.items():
                        info_to_print += str(item).ljust(len_cell - 2).rjust(len_cell) + '|'

            print(info_to_print)
            print(''.ljust(len(info_to_print), '-'))
            

# *** IMAGES ***
def create_im_path(im_id, im_set, db_absolute_path = ABSOLUTE_PATH):
    """
    This function create the absolute path of an image
    
    Argument:
    im_id            -- int corresponding to the image id in the kitti dataset
    im_set           -- 'train' or 'test'
    db_absolute_path -- absolute path to the Kitti root folder
    
    Returns:
    im_path -- absolute path to the image
    """
    len_filename = 6
    
    im_filename = str(im_id).zfill(len_filename) + ".png"
    
    im_path = db_absolute_path + globals()[(im_set +'_path_im').upper()] + im_filename
    
    return im_path

def import_im(im_id, im_set, db_absolute_path = ABSOLUTE_PATH):
    """
    This import the image in a numpy array
    
    Argument:
    im_id            -- int corresponding to the image id in the kitti dataset
    im_set           -- 'train' or 'test'
    db_absolute_path -- absolute path to the Kitti root folder
    
    Returns:
    im -- image in a numpy array
    """
    # Create file absolute path
    im_path = create_im_path(im_id, im_set, db_absolute_path)
    
    # Import the image
    im = misc.imread(im_path)
    
    return im

def create_boxes(labels, types_to_display = DEFAULT_TYPES_TO_DISPLAY):
    """
    This function create boxes according to a dictionnary of labels
    
    Argument:
    labels            -- list of dictionnaries containing the spacial information
    types_to_display  -- list of the type of object to display
    
    Returns:
    boxes_to_display  -- list of patches.Rectangle objects
    """
    
    boxes_to_display = []
    
    for obj in labels:
        if obj['type'] in types_to_display:
            bbox = obj['bbox']
            boxes_to_display.append(
                patches.Rectangle(
                    (bbox['x_min'], bbox['y_min']),        # (x,y)
                    bbox['x_max'] - bbox['x_min'],         # width
                    bbox['y_max'] - bbox['y_min'],         # height
                    obj['alpha'],                          # rotation angle
                    linewidth = 3,                         # linewidth
                    edgecolor = COLOR_TYPE[obj['type']],   # color corresponding to type
                    facecolor = 'none'                     # not fill
                )
            ) 
            
    return boxes_to_display

def display_im(im_id, im_set, display_boxes = True, display_info = True, 
               types_to_display = DEFAULT_TYPES_TO_DISPLAY, 
               info_to_display = DEFAULT_INFO_TO_DISPLAY, db_absolute_path = ABSOLUTE_PATH, 
               im_width = FIG_WIDTH, im_height = FIG_HEIGHT, im_axis='off', display_title = True):
    """
    This function displays an image from its id
    
    Argument:
    im_id            -- int corresponding to the image id in the kitti dataset
    im_set           -- 'train' or 'test'
    display_boxes    -- True or False
    display_info     -- True or False
    types_to_display -- list of the name of the types of object to consider
    info_to_display  -- list of the name of the information to display
    db_absolute_path -- absolute path to the Kitti root folder
    im_width         -- width of the image to display
    im_height        -- height of the image to display
    im_axis          -- 'on' or 'off'
    display_title    -- True or False to display the id of the image as the title of the subplot
    
    Returns:
    Display image
    """
    # Import the image 
    im = import_im(im_id, im_set, db_absolute_path)

    # Display image
    fig, ax = plt.subplots(1, figsize=(im_width, im_height))
    
    if display_title == True:
        ax.set_title(im_set +'_'+ str(im_id), fontsize = FIG_FONT_SIZE_TITLE)
    
    ax.imshow(im)
    ax.axis(im_axis)
    
    if display_boxes == True or display_info == True:
        # Get the labels of an image
        labels = import_labels(im_id, im_set, db_absolute_path)
    
    # Display boxes
    if display_boxes == True:        
        # Get the list of boxes
        boxes = create_boxes(labels, types_to_display)
        
        # Add the boxes to the picture
        for box in boxes:
            ax.add_patch(box)

    plt.show()
    
    # Display information
    if display_info == True:
        # Display information about the object
        print_labels(labels, types_to_display, info_to_display)
        
# *** FILES ***
def get_data_list(im_set, db_absolute_path = ABSOLUTE_PATH):
    """
    This function return a numpy array of all the ids of the files in a folder
    
    Argument:
    im_set           -- 'train' or 'test'
    db_absolute_path -- absolute path to the Kitti root folder
    
    Returns:
    label_path -- absolute path to the label
    """
    folder_path = db_absolute_path + globals()[(im_set +'_path_im').upper()]
    
    list_files = [f for f in listdir(folder_path) if (isfile(join(folder_path, f)) and f.endswith('.png'))]
    
    list_ids = [int(file_name[:-4]) for file_name in list_files]
    
    return np.array(sorted(list_ids))

