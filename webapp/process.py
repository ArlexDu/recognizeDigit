from PIL import Image
import numpy as np

# 转化灰度图片
def convertGray(img,width,height):
    dest = Image.new('L', (width, height), 255)
    for i in range(width):
        for j in range(height):
            r,g,b,a = img.getpixel((i,j))
            brightness = int(a)
            dest.putpixel((i,j),brightness)
    return dest

# 获取分割的图片
def getCutImage(img,data,width,height):
    data = np.reshape(data, (height, width))
    print(data.shape)
    zeroh = np.zeros((height,1))
    widths = []
    judge = True
    for i in range(width):
        vh = data[:,i]
        if (vh == zeroh).all():
            judge = True
        else:
            if judge:
                widths.append(i)
                judge = False
    if widths[-1] != (width-1):
        widths.append(width-1)
    widths.sort()
    w1s = []
    w2s = []
    h1s = []
    h2s = []
    for w1,w2 in zip(widths[:-1],widths[1:]):
        img_tmp = img.crop((w1,0,w2,height))
        bbox = img_tmp.getbbox()
        w1 = w1 + bbox[0]
        w2 = w1 + bbox[2]
        h1 = bbox[1]
        h2 = bbox[3]
        w1s.append(w1)
        w2s.append(w2)
        h1s.append(h1)
        h2s.append(h2)
    return w1s,w2s,h1s,h2s

def process(image):
    img = Image.open(image)
    width, height = img.size
    gray_img = convertGray(img, width, height)
    data = np.matrix(gray_img.getdata())
    w1s,w2s,h1s,h2s = getCutImage(gray_img, data, width, height)
    arrays = []
    for w1,w2,h1,h2 in zip(w1s,w2s,h1s,h2s):
        a = data_process(img,w1,w2,h1,h2)
        arrays.append(a)
    return arrays


# 处理函数
def data_process(img,w1,w2,h1,h2):
    widthlen = w2 - w1
    heightlen = h2 - h1
    print(widthlen,heightlen)
    if heightlen > widthlen:
        widthlen = int(20.0 * widthlen/heightlen)
        heightlen = 20
    else:
        heightlen = int(20.0 * widthlen/heightlen)
        widthlen = 20

    hstart = int((28 - heightlen) / 2)
    wstart = int((28 - widthlen) / 2)

    img_temp = img.crop((w1,h1,w2,h2)).resize((widthlen, heightlen), Image.NEAREST)
    new_img = Image.new('L', (28,28), 255)
    new_img.paste(img_temp, (wstart, hstart),mask = img_temp)
    # new_img.show()
    imgdata = list(new_img.getdata())
    img_array = np.array([(255.0 - x) / 255.0 for x in imgdata])
    return img_array



