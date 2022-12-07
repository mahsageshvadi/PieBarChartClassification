import numpy as np
import cv2
import math
import argparse

from config import Config, MakeDir, ClearDir, RemoveDir



config = Config()

parser = argparse.ArgumentParser()
parser.add_argument("--nimages", default = config.number_of_images, type = int)
parser.add_argument("--ttv", default = 'Train')
parser.add_argument("--save", default = True)

a = parser.parse_args()

number_of_images = a.nimages
number_of_piechart_images = number_of_images//2
number_of_barchart_images = number_of_images//2


if a.ttv == 'Train':
    save_dir = config.train_data_dir
    
elif a.ttv == 'Test':
    save_dir = config.test_data_dir
    
elif a.ttv == 'Validation':
    save_dir = config.validation_data_dir



def Normalize(arr):
    return arr / np.sum(arr)
    
    

def generate_barchart(number_of_barchart_images):
    
    barchart_images = []
    
    for i in range(number_of_barchart_images):
    
        image_log_for_statistics = {}
        
        if config.number_of_channels == 3:
            colors = np.random.uniform(0.0, 0.9,size = (config.max_obj_num_for_bar,3))

        image = np.ones(shape=(config.image_width, config.image_height, config.number_of_channels))
        
        number_of_bars = np.random.randint(2, config.max_obj_num_for_bar + 1)
        thickness = np.random.randint(1, config.max_thickness)
        padding = np.random.randint(0, 50)
        ratio_for_padding = (config.image_width - 2* padding)/ config.image_width
        height = np.random.randint(10, config.image_height * ratio_for_padding, size = number_of_bars)
       
        image_log_for_statistics['heigth'] = height
        image_log_for_statistics['number_of_bars'] = number_of_bars

        
        barWidth = int( ((config.image_width-3*(number_of_bars+1)-3)//number_of_bars * (np.random.randint(50,100)/100.0))* ratio_for_padding )
        barWidth = max(barWidth, 4)
        spaceWidth = int((config.image_width-(barWidth)*number_of_bars)//(number_of_bars+1) * ratio_for_padding)

        sx = (config.image_width - barWidth*number_of_bars - spaceWidth*(number_of_bars-1))//2
        
        
        for j in range(number_of_bars):

            sy = int(config.image_width * ratio_for_padding)
            ex = sx + barWidth
            ey = sy - height[j]
            
            if config.number_of_channels == 3:
                cv2.rectangle(image,(sx,sy),(ex,ey),colors[j],-1)
            else:
                cv2.rectangle(image,(sx,sy),(ex,ey),0,thickness)

            sx = ex + spaceWidth
        noises = np.random.uniform(0, 0.05, (config.image_width, config.image_height,3))
        image = image + noises
        _min = 0.0
        _max = image.max()
        image -= _min
        image /= (_max - _min)

        barchart_images.append(image)
        
        if a.save:
            cv2.imwrite( save_dir+ '/Bar_image{}.jpg'.format(i), image* 255)

        

    return barchart_images
    
def generate_piechart(number_of_piechart_images):
        

        piechart_images = []
        for i in range(number_of_piechart_images):
            
            number_of_pies = np.random.randint(2, config.max_obj_num_for_pie + 1)

            max_w_h = max(config.image_width, config.image_height)/2
            
            r = np.random.randint(0.2* max_w_h,max_w_h -20)
            thickness = np.random.randint(1,3)
            
            colors = np.random.uniform(0.0, 0.9,size = (config.max_obj_num_for_pie,3))
            center = (int(config.image_width/2),int(config.image_height/2))
            image = np.ones(shape=(config.image_width, config.image_height, 3))

            angles = Normalize(np.random.randint(10,60,size=(number_of_pies)))

            start_angle = 90 - np.random.randint(0,360*angles[0])/2.0
            _cur_start_angle = start_angle
            # cv2.circle(image,center,r,0,thickness)

            for j in range(number_of_pies):
                _cur_end_angle = _cur_start_angle + angles[j] * 360.0

                cv2.ellipse(image, center, (r, r), 270, -_cur_start_angle, -_cur_end_angle, colors[j], -1)
                _cur_start_angle = _cur_end_angle

            noises = np.random.uniform(0, 0.05, (config.image_width, config.image_height,3))
            image = image + noises

            _min = 0.0
            _max = image.max()
            image -= _min
            image /= (_max - _min)

            piechart_images.append(image)
            if a.save:
            
                cv2.imwrite( save_dir + '/Pie_image{}.jpg'.format(i), image *255)

                    
        return piechart_images


def generate_data():

    barchart_images = generate_barchart(number_of_barchart_images)
    piechart_images = generate_piechart(number_of_piechart_images)


        
    return barchart_images, piechart_images





if __name__ == '__main__':


    MakeDir(config.base_dir)
    if a.ttv == 'Train':
        ClearDir(config.train_data_dir)
    if a.ttv == 'Validation':
        ClearDir(config.validation_data_dir)
    if a.ttv == 'Test':
        ClearDir(config.test_data_dir)
    generate_data()
    
