import cv2
import numpy as np
#creating a background
cap = cv2.VideoCapture(0) 


while cap.isOpened():
    ret, background = cap.read() 
    if ret:
        cv2.imshow("image", background)
        if cv2.waitKey(5) == ord('q'):
            #save the background image
            cv2.imwrite("resources/background.png", background)
            break
cap.release()
cv2.destroyAllWindows()

# Now making the invisibility cloak


cap = cv2.VideoCapture(0)
background = cv2.imread('resources/background.png')

while cap.isOpened():
    #capturing the live frame
    ret, live = cap.read()
    if ret:
        #converting from rgb to hsv color space
        hsv_frame = cv2.cvtColor(live, cv2.COLOR_BGR2HSV)

        #range for lower red
        lower_red = np.array([0,120,170])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)

        #range for lower red
        lower_red = np.array([170,120,70])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)

        #generating the final red mask cause both are numpy arrays
        red_mask = mask1 + mask2

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 10) 
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations = 1)  

        #subsituting the red portion with backgrpound image
        mask1 = cv2.bitwise_and(background, background, mask= red_mask)
        
        # detecting things which are not red
        red_free = cv2.bitwise_not(red_mask)

        # if cloak is not present show the current image
        mask2 = cv2.bitwise_and(live, live, mask= red_free)

        
        #final output
        cv2.imshow("cloak", cv2.resize((mask1 + mask2),(0,0),fx=1.5,fy=1.5))
        if cv2.waitKey(5) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()