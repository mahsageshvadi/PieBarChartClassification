import numpy as np
import cv2
import math


from config import Config

config = Config()


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
        height = np.random.randint(10, config.image_height - 20, size = number_of_bars)
       
        image_log_for_statistics['heigth'] = height
        image_log_for_statistics['number_of_bars'] = number_of_bars

        
        barWidth = int( (config.image_width-3*(number_of_bars+1)-3)//number_of_bars * (np.random.randint(50,100)/100.0) )
        barWidth = max(barWidth, 4)
        spaceWidth = (config.image_width-(barWidth)*number_of_bars)//(number_of_bars+1)

        sx = (config.image_width - barWidth*number_of_bars - spaceWidth*(number_of_bars-1))//2
        
        
        for i in range(number_of_bars):

            sy = config.image_width - 1
            ex = sx + barWidth
            ey = sy - height[i]
            
            if config.number_of_channels == 3:
                cv2.rectangle(image,(sx,sy),(ex,ey),colors[i],-1)
            else:
                cv2.rectangle(image,(sx,sy),(ex,ey),0,thickness)

            sx = ex + spaceWidth
        noises = np.random.uniform(0, 0.05, (size, size,3))
        image = image + noises
        _min = 0.0
        _max = image.max()
        image -= _min
        image /= (_max - _min)

        barchart_images.append(image)

    return barchart_images
    
def generate_piechart(number_of_piechart_images):
        

        piechart_images = []
        for i in range(number_of_piechart_images):
            
            number_of_pies = np.random.randint(2, config.max_obj_num_for_pie + 1)

            r = np.random.randint(25,45)        # Radii of the pie. (random)

            
            colors = np.random.uniform(0.0, 0.9,size = (config.max_obj_num_for_pie,3))
            center = (int(config.image_width/2),int(config.image_height/2))  #
            image = np.ones(shape=(config.image_width, config.image_height, 3))
            subImages = [np.ones(shape=(config.image_width, config.image_height,3)) for i in range(6)]
            angles = Normalize(np.random.randint(10,60,size=(number_of_pies)))

            start_angle = 90 - np.random.randint(0,360*angles[0])/2.0
            _cur_start_angle = start_angle
            # cv2.circle(image,center,r,0,thickness)

            for i in range(number_of_pies):
                _cur_end_angle = _cur_start_angle + angles[i] * 360.0

                cv2.ellipse(image, center, (r, r), 270, -_cur_start_angle, -_cur_end_angle, colors[i], -1)
                _cur_start_angle = _cur_end_angle

            noises = np.random.uniform(0, 0.05, (size, size,3))
            image = image + noises

            _min = 0.0  # because the image is not 0/1 black-and-white image, is a RGB image.
            _max = image.max()
            image -= _min
            image /= (_max - _min)

            piechart_images.append(image)
                    
        return piechart_images


def generate_data():

    piechart_images = generate_piechart(config.number_of_piechart_images)
    barchart_images = generate_barchart(config.number_of_barchart_images)
    
    
    





    for i in range(config.number_of_piechart_images):
        cv2.imwrite('/Users/mahsa/Documents/PhDlife/Courses/1st_Semester/ML/ML_Final/Data/Pie_image{}.jpg'.format(i), piechart_images[i] * 255)
    
    for i in range(config.number_of_barchart_images):

        cv2.imwrite('/Users/mahsa/Documents/PhDlife/Courses/1st_Semester/ML/ML_Final/Data/Bar_image{}.jpg'.format(i), barchart_images[i] * 255)

generate_data()
     
     

