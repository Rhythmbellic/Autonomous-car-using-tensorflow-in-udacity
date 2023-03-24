#importing libraries important for processing images
import cv2
import numpy as np
import matplotlib.pyplot as plt
#reading the image
image=cv2.imread('test_image.jpg')
#making a copy so that real image do not change
lane_image=np.copy(image)
def canny(Image):
  gray=cv2.cvtColor(Image,cv2.COLOR_RGB2GRAY)   #changing image to grayscale
  blur=cv2.GaussianBlur(gray,(5,5),0)           #using gaussian blur to remove noise
  canny=cv2.Canny(blur,50,150)                  #using canny to smoothen the image
  return canny
def see(ab,Image):                              # creating function to see the image
  cv2.imshow(ab,Image)
  cv2.waitKey(0)
#making a polygon mask (triangle mask) of required region
def region_of_intrest(image):
  height=image.shape[0]
  polygons=np.array([[(200,height),(1100,height),(550,250)]])
  mask=np.zeros_like(image)
  cv2.fillPoly(mask,polygons,255)
  masked_image = cv2.bitwise_and(image,mask)
  return masked_image
#plotting the lines
def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),4)
    return line_image
#now making 2 function to optimize the lines
def make_cordinates(image,line_parameters):
    slope, intercept=line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])
def avg_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2 =line.reshape(4)
        parameter=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameter[0]
        intercept=parameter[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    left_line=make_cordinates(image,left_fit_average)
    right_line=make_cordinates(image,right_fit_average)
    return np.array([left_line,right_line])
# no we have created the function we will now implement it

#for the image
a=canny(lane_image)
plt.imshow(a)
crop=region_of_intrest(a)
lines=cv2.HoughLinesP(crop,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
averaged_lines=avg_slope_intercept(lane_image,lines)
line_image=display_lines(lane_image,averaged_lines)
combo_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
see("lines",combo_image)


# for videos
cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame=cap.read()
    a=canny(frame)
    crop=region_of_intrest(a)
    lines=cv2.HoughLinesP(crop,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    averaged_lines=avg_slope_intercept(frame,lines)
    line_image=display_lines(frame,averaged_lines)
    combo_image=cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow('result',combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
