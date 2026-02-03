import cv2
from matplotlib import pyplot as plt

img = cv2.imread('roads/road1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

car_data = cv2.CascadeClassifier('haar/haarcascade_cars.xml')
stop_data = cv2.CascadeClassifier('haar/haarcascade_stopsign.xml')

car_coords = []
stop_coords = []

try:
    car_coords = car_data.detectMultiScale(img_gray, minSize=(20, 20)).tolist()
    print('car_coords:', car_coords)
except:
    print('No cars found')

try:
    stop_coords = stop_data.detectMultiScale(img_gray, minSize=(20, 20), scaleFactor=1.1, minNeighbors=5).tolist()
except:
    print('No stopsigns found')

img_height, img_width, img_channels = img.shape
left_border =img_width/2
right_border = img_width

def check_forward(car_coords, stop_coords):
    if len(stop_coords) != 0:
        for (x, y, width, height) in car_coords:
            cv2.circle(img_rgb, (x + (width // 2), y + height // 2), int(height // 2), (0, 255, 75), 7)
        for (x, y, width, height) in stop_coords:
            cv2.circle(img_rgb, (x + (width // 2), y + height // 2), int(height // 2), (0, 255, 75), 5)
        return False
    elif len(car_coords) == 0:
        return True
    else:
        for (x, y, width, height) in car_coords:
            if x > left_border and x + width < right_border:
                if width / img_width < 0.15:
                    return False
        return True

print(check_forward(car_coords, stop_coords))

plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()


