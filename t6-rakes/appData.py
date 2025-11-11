def showImage(caption, previewImage, sv=False):
    cv.imshow(caption, previewImage)
    cv.waitKey(0)
    cv.destroyAllWindows()
    if (sv):
        cv.imwrite('d:/'+caption+'.jpg', previewImage)
        # endif

def image1banding1(image):
    height, width = image.shape[:2]
    new_size = min(height, width)
    img_resize = cv.resize(image, (new_size, new_size))
    return img_resize

def dariMedium(img_path):
    # loadImage
    pil_image = Image.open(image_path).convert('RGB')
    image = np.array(pil_image)
    preview = cv.resize(image, (1024, 1024))
    # showImage('downScaleForPreview', preview)

    # edgePreservingFiltering
    filtered = cv.bilateralFilter(preview, d=9, sigmaColor=75, sigmaSpace=75)
    mask = cv.inRange(filtered, (0,0,0), (50,50,50))
    showImage('quickAndDirty_manualMaskForDarkSpots', mask)

    inpainted = cv.inpaint(filtered, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA) #cv.INPAINT_NS
    inpaintedImage = cv.cvtColor(inpainted, cv.COLOR_BGR2RGB)
    showImage('restoredImage', inpaintedImage)

def deleteGambar():
    for ffolder in ['inputImage', 'outputImage']:
        for fname in os.listdir(os.path.join(os.getcwd(), 'static', ffolder)):
            os.remove(os.path.join(os.getcwd(), 'static', ffolder, fname))
            # endfor
        # endfor
    return 'dataBerhasilDihapusBoss...'

def listImage():
    listOutputImage = [fname for fname in os.listdir(os.path.join(os.getcwd(), 'static', 'outputImage'))]
    if (len(listOutputImage) < 11):
        listFix = sorted([listOutputImage[i] for i in range(0,len(listOutputImage))], reverse=True)
        # endif
    elif (len(listOutputImage) > 10)and(len(listOutputImage) < 21):
        a = sorted([listOutputImage[i] for i in range(0,10)], reverse=True)
        b = sorted([listOutputImage[i] for i in range(10,len(listOutputImage))], reverse=True)
        listFix = a+b
        # endif
    elif (len(listOutputImage) > 20)and(len(listOutputImage) < 31):
        a = sorted([listOutputImage[i] for i in range(0,10)], reverse=True)
        b = sorted([listOutputImage[i] for i in range(10,20)], reverse=True)
        c = sorted([listOutputImage[i] for i in range(20,len(listOutputImage))], reverse=True)
        listFix = a+b+c
        # endif
    else:
        deleteGambar()
        listFix = os.listdir(os.path.join(os.getcwd(), 'static', 'outputImage'))
        # endif
    return listFix

if (__name__=='__main__'):
    from PIL import Image, ImageFilter
    import numpy as np
    import cv2 as cv
    import os

    # image_path = os.path.join('d:/','images/faces', 'face_khaidir.jpg')
    # image_path = os.path.join('d:/', 'images', 'image3.png')
    # dariMedium(image_path)

    print(listImage())
