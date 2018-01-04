from PIL import Image
import numpy as np
def process(image):
    img = Image.open(image)
    bbox = Image.eval(img,lambda px:255-px).getbbox()
    if bbox == None:
        return None
    widthlen = bbox[2] - bbox[0]
    heightlen = bbox[3] - bbox[1]
    print (widthlen,heightlen)   
    if heightlen > widthlen:
        widthlen = int(20.0 * widthlen/heightlen)
        heightlen = 20
    else:
        heightlen = int(20.0 * widthlen/heightlen)
        widthlen = 20

    hstart = int((28 - heightlen) / 2)
    wstart = int((28 - widthlen) / 2)

    img_temp = img.crop(bbox).resize((widthlen, heightlen), Image.NEAREST)
    new_img = Image.new('L', (28,28), 255)
    new_img.paste(img_temp, (wstart, hstart),mask = img_temp)
    # new_img.show()
    imgdata = list(new_img.getdata())
    img_array = np.array([(255.0 - x) / 255.0 for x in imgdata])
    return img_array

# process('/Users/arlex/Documents/Project/Webapp/recognizeDigit/media/1515083987348.png')
