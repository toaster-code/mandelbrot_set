import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext
from matplotlib.widgets import RectangleSelector, TextBox, Button

# Define the complex plane
xmin, xmax, ymin, ymax = -2, 1, -1, 1
width, height = 256, 256

# Define the maximum number of iterations and threshold value
max_iter = 30
threshold = 2
threshold_squared = threshold*threshold

# Define the function to update the precision value
def update_precision(text, ax):
    global precision
    precision = int(text)
    update_image(ax, xmin, xmax, ymin, ymax, precision)
    ax.title.set_text('Precision: {}'.format(precision))


#define the precision method (None = normal, >=28 = high)
precision = 10

# Update the image with the new precision value
def update_image(ax, xmin, xmax, ymin, ymax, precision=None):
    if precision is None:
        return update_image_normal_precision(ax, xmin, xmax, ymin, ymax)
    else:
        return update_image_high_precision(ax, xmin, xmax, ymin, ymax, precision)


def update_image_normal_precision(ax, xmin, xmax, ymin, ymax):
    # Create a numpy array of zeros
    image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # Convert pixel coordinates to complex number
            z = complex(xmin + (xmax - xmin) * x / width, ymin + (ymax - ymin) * y / height)
            c = z

            # Apply Mandelbrot set formula
            for i in range(max_iter):
                if abs(z) > threshold:
                    break
                z = z ** 2 + c

            # Map iteration values to colors
            color = (i * 255 // max_iter,)*3

            # Set the pixel value in the image
            image[y, x] = color[0]

    # Create an image from the pixel values
    img = Image.fromarray(image)

    # Display the image using Matplotlib
    ax.imshow(img, cmap='jet')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    plt.draw()


def update_image_high_precision(ax, xmin, xmax, ymin, ymax, precision):
    # Set the desired precision for decimal calculations
    getcontext().prec = precision

    # Create a numpy array of zeros
    image = np.zeros((height, width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # Convert pixel coordinates to complex number using Decimal
            x_decimal = Decimal(str(xmin)) + (Decimal(str(xmax)) - Decimal(str(xmin))) * Decimal(str(x)) / Decimal(str(width))
            y_decimal = Decimal(str(ymin)) + (Decimal(str(ymax)) - Decimal(str(ymin))) * Decimal(str(y)) / Decimal(str(height))
            zre = x_decimal
            zim = y_decimal
            cre = zre
            cim = zim

            # Apply Mandelbrot set formula using Decimal
            for i in range(max_iter):
                if (zre*zre +zim*zim) > threshold_squared:
                    break
                z2re = zre*zre -zim*zim
                z2im = 2*zre*zim

                zre = z2re + cre
                zim = z2im + cim


            # Map iteration values to colors
            color = (i * 255 // max_iter,)*3

            # Set the pixel value in the image
            image[y, x] = color[0]

    # Create an image from the pixel values
    img = Image.fromarray(image)

    # Display the image using Matplotlib
    ax.imshow(img, cmap='jet')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    plt.draw()

def onselect(eclick, erelease):
    global xmin, xmax, ymin, ymax
    if eclick.ydata > erelease.ydata:
        eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
    if eclick.xdata > erelease.xdata:
        eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
    # Calculate the new bounds of the complex plane based on the selected rectangle
    x_range = xmax - xmin
    y_range = ymax - ymin
    new_xmin = xmin + eclick.xdata * x_range / width
    new_xmax = xmin + erelease.xdata * x_range / width
    new_ymin = ymin + eclick.ydata * y_range / height
    new_ymax = ymin + erelease.ydata * y_range / height
    # Update the image with the new bounds
    update_image(ax, new_xmin, new_xmax, new_ymin, new_ymax, precision=precision)
    xmin, xmax, ymin, ymax = new_xmin, new_xmax, new_ymin, new_ymax

    # Clear the rectangle selector
    # rect_selector.set_active(False)
    rect_selector.set_visible(False)
    # rect_selector.update()
    # rect_selector.disconnect_events()

fig, ax = plt.subplots()
update_image(ax, xmin, xmax, ymin, ymax, precision=precision)

# Add a TextBox widget for entering precision value
precision_ax = plt.axes([0.1, 0.05, 0.2, 0.075])
precision_box = TextBox(precision_ax, 'Precision:', initial='14')
precision_box.on_submit(lambda x: update_precision(int(x), ax))

# Add a "Confirm" button to update the precision value
confirm_ax = plt.axes([0.35, 0.05, 0.1, 0.075])
confirm_button = Button(confirm_ax, 'Confirm')
confirm_button.on_clicked(lambda x: update_precision(int(precision_box.text), ax))


# Add rectangle selector
rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1],
                                   minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)


plt.show()
