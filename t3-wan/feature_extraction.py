def show(hasil):
    cv.imshow('hasil',hasil)
    cv.waitKey(0)
    cv.destroyAllWindows()

def SIFT(path):
    original_face = cv.resize(cv.imread(path), (128,128))
    gray_face = cv.cvtColor(original_face, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    original_keypoints, original_descriptor = sift.detectAndCompute(gray_face, None)
    keypoints_with_size = np.copy(original_face)
    print('*SIFT feature')
    print(original_descriptor.shape)
    hist, bins = np.histogram(original_descriptor, bins=4)
    print('hist:', hist)
    print('bins:', bins)
    imgSIFT = cv.drawKeypoints(original_face, original_keypoints, keypoints_with_size, flags= cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show(imgSIFT)

def FAST(path):
    original_face = cv.resize(cv.imread(path), (128,128))
    gray_face = cv.cvtColor(original_face, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    fast.setThreshold(30)
    fast.setNonmaxSuppression(False)
    keypoints_without_nonmax = fast.detect(gray_face, None)
    image_without_nonmax = np.copy(gray_face)
    imgFAST = cv.drawKeypoints(gray_face, keypoints_without_nonmax, image_without_nonmax, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print('*FAST feature')
    print(len(keypoints_without_nonmax))
    show(imgFAST)

def ORB(path):
    original_face = cv.resize(cv.imread(path), (128,128))
    gray_face = cv.cvtColor(original_face, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    original_keypoints, original_descriptor = orb.detectAndCompute(gray_face, None)
    keypoints_without_size = np.copy(original_face)
    imgORB = cv.drawKeypoints(original_face, original_keypoints, keypoints_without_size, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print('*ORB feature')
    print(original_descriptor.shape)
    hist, bins = np.histogram(original_descriptor, bins=4)
    print('hist:', hist)
    print('bins:', bins)
    show(imgORB)

def SLIC(img):
    # convert_to_lab_color_space
    lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    # initialize_SLIC
    slic = cv.ximgproc.createSuperpixelSLIC(lab_img,  region_size=25, algorithm=cv.ximgproc.SLICO)
    # perform_iterations
    slic.iterate()
    # get_the_superpixels_label
    labels = slic.getLabels()
    # get_the_number_of_superixels
    num_superpixels = slic.getNumberOfSuperpixels()
    # create_an_overlay_of_superpixel_boundaries
    mask = slic.getLabelContourMask(True)
    overlay = np.copy(img)
    overlay[mask != 0] = [0, 255, 0] # green_boundaries

    # display_result
    show('originalImage', img)
    show('superpixel_overlay', overlay)

def slic_segmetation(img, region_size=25, ruler=10.0):
    show_plt('original_face', img)
    slic = cv.ximgproc.createSuperpixelSLIC(img, region_size=region_size, ruler=ruler)
    slic.iterate(10)
    labels = slic.getLabels()
    segmented_img = np.zeros_like(img)
    for label in np.unique(labels):
        mask = labels == label
        segmented_img[mask] = np.mean(img[mask], axis=0)

    slic_img = segmented_img.astype(np.uint8)
    show_plt('slic_img', slic_img)
    print(slic_img.shape)
    slic_img_gray = cv.cvtColor(slic_img, cv.COLOR_BGR2GRAY)
    show_plt('slic_img_gray', slic_img_gray)
    print(slic_img_gray.shape)
    hist, bins = np.histogram(slic_img_gray, bins=4)
    print('hist', hist)
    print('bins', bins)
    return hist

def PCA(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img_flat = img.reshape(1, -1).astype(np.float32)
    # compute pca
    mean, eigenvectors = cv.PCACompute(img_flat, mean=None, maxComponents=500)
    # project data onto principal components
    projected_img = cv.PCAProject(img_flat, mean, eigenvectors)
    # recontruct the original data
    reconstructed_data = cv.PCABackProject(projected_img, mean, eigenvectors)
    print('original data:')
    print(img)
    print('projected_img:')
    print(projected_img)
    print('reconstructed_data:')
    print(reconstructed_data)

def PCA2(path):
    img = cv.resize(cv.imread(path, cv.IMREAD_GRAYSCALE), (128,128))
    print(img, img.shape)

    # create a PCA object with 4 principal components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=4)
    # pca implementation on img
    pca_attributes = pca.fit_transform(img)
    # see the variance of each attribute
    print(pca.explained_variance_ratio_)

if __name__=='__main__':
    import cv2 as cv
    import numpy as np

    # SIFT('d:/images/faces/face_monalisa.jpg')
    # FAST('d:/images/faces/face_monalisa.jpg')
    # ORB('d:/images/faces/face_monalisa.jpg')
    PCA2('d:/images/faces/face_monalisa.jpg')
