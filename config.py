import os
import shutil

class Config:

    
    image_width = 100
    image_height = 100
    number_of_channels = 3
    max_obj_num_for_bar = 6
    min_obj_num = 3
    max_thickness = 3
            
    #pie specific
    max_obj_num_for_pie = 10
    
    number_of_images = 1000
    
    #Dir
    
    base_dir = os.path.abspath('./Data')
    train_data_dir = base_dir + '/TrainData'
    validation_data_dir = base_dir + '/ValidationData'
    test_data_dir = base_dir + '/TestData'
    
    
def ClearDir(path):
        if os.path.exists(path):
            shutil.rmtree(path=path)
        os.mkdir(path)

def MakeDir(path):
        if not os.path.exists(path):
            os.mkdir(path)

def RemoveDir(path):
        if os.path.exists(path):
            os.remove(path)

