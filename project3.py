import cv2 
from matplotlib import pyplot as plt 
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\Sins\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe"

image_file =cv2.imread("download.png")


#histogram
def histo(x ,image):
    histogram = image.flatten()
    plt.hist(histogram, bins=256, range=(0, 256), color='gray')
    plt.title(x)
    plt.show()


def show( x ,image_file ):
    # Resize the image
    scale_percent = 70  # percent of original size
    width = int(image_file.shape[1] * scale_percent / 100)
    height = int(image_file.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(image_file, dim, interpolation=cv2.INTER_AREA)
    # Show the resized image
    cv2.imshow(x, resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img =cv2.imread("download.png")
show("original image", img)

#create pic to string and print 
custom_config = r'-l eng --oem 3 --psm 6' 
text = pytesseract.image_to_string(image_file,config=custom_config)
print(text)

#showing histogram
histo("orginal image" , img)

#start editing pic

#invert a picture
inver = cv2.bitwise_not(img)
cv2.imwrite("inverted.jpg" , inver)
show("inverted", inver)


#create a grey scale version of image
def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray_image = gray(img)
cv2.imwrite("grey.jpg" , gray_image)
show("grey_scaled", gray_image)
histo("graysclaed image" , gray_image)

#Create Black and white
thresh , im_bw = cv2.threshold(gray_image, 210 ,220 , cv2.THRESH_BINARY)
cv2.imwrite("bandw.jpg" ,im_bw )
show("black and white", im_bw)
histo("threshold" , im_bw)


# remove noise
def remove_nosie(image):
    import numpy as np
    kernal = np.ones((1,1) , np.uint8)
    image = cv2.dilate(image , kernal , iterations  = 1)
    kernal = np.ones((1,1),np.uint8)
    image = cv2.erode(image , kernal , iterations = 1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN ,kernal)
    image = cv2.medianBlur(image , 3 )
    return(image)

no_noise = remove_nosie(im_bw)
cv2.imwrite("no_noise.jpg" ,no_noise)
show("Removed noise", no_noise)


#showing histogram
histo( "After removal of noises and threshold " , no_noise )

custom_config = r'-l eng --oem 3 --psm 6' 
text = pytesseract.image_to_string(no_noise,config=custom_config)
print("-----------no noise-----------")
print(text)

def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)
eroded_image1 = thin_font(no_noise)
cv2.imwrite("eroded_image.jpg", eroded_image1)
show("eroded image " , eroded_image1)


custom_config = r'-l eng --oem 3 --psm 6' 
text = pytesseract.image_to_string(eroded_image1,config=custom_config)
print("---------- after thin text ------------")
print(text)


def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

eroded_image = thick_font(no_noise)
cv2.imwrite("thick_image2.jpg", eroded_image)
show("dilated image" , eroded_image)

custom_config = r'-l eng --oem 3 --psm 6' 
text = pytesseract.image_to_string(eroded_image,config=custom_config)
print("---------- Text after corrections ------------")
print(text)
