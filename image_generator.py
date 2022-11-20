import numpy as np
import cv2
import math


from config import Config

config = Config()
number_of_barchart_images = 20



def Normalize(arr):
    return arr / np.sum(arr)
    
    

def generate_barchart():
    
    
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

    return image
    
def generate_piechart():
        r = np.random.randint(25,45)        # Radii of the pie. (random)
        thickness = np.random.randint(1,3)  # line thickness.   (random)
        center = (int(config.image_width/2),int(config.image_height/2))
        image = np.ones(shape=(config.image_width, config.image_height, 1))
        
        number_of_pie_slots = np.random.randint(2, config.max_obj_num_for_pie + 1)

        angles = Normalize(np.random.randint(10,60,size=(number_of_pie_slots)))


        
        start_angle = 90 - np.random.randint(0,360*angles[0])/2.0
        _cur_start_angle = start_angle
        cv2.circle(image,center,r,0,thickness)
        
        for i in range(number_of_pie_slots):
                _cur_end_angle = _cur_start_angle + angles[i] * 360.0
                cv2.line(image,center, (50-int(r*math.sin(math.pi*_cur_start_angle/180.0)),
                                50-int(r*math.cos(math.pi*_cur_start_angle/180.0))),0,thickness)
                _cur_start_angle = _cur_end_angle
                
        return image



    


images = []
for i in range(number_of_barchart_images):

 images.append(generate_piechart())

for i in range(number_of_barchart_images):
    cv2.imwrite('/Users/mahsa/Documents/PhDlife/Courses/1st_Semester/ML/ML_Final/image{}.jpg'.format(i), images[i] * 255)


     
     

