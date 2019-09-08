## Iris Recognition

is an automated method of biometric identification that uses mathematical pattern-recognition techniques on video images of one or both of the irises of an individual's eyes, whose complex patterns are unique, stable, and can be seen from some distance.

Retinal scanning is a different, ocular-based biometric technology that uses the unique patterns on a person's retina blood vessels and is often confused with iris recognition. Iris recognition uses video camera technology with subtle near infrared illumination to acquire images of the detail-rich, intricate structures of the iris which are visible externally. Digital templates encoded from these patterns by mathematical and statistical algorithms allow the identification of an individual or someone pretending to be that individual.[1] Databases of enrolled templates are searched by matcher engines at speeds measured in the millions of templates per second per (single-core) CPU, and with remarkably low false match rates.

Several hundred million persons in several countries around the world have been enrolled in iris recognition systems for convenience purposes such as passport-free automated border-crossings and some national ID programs. A key advantage of iris recognition, besides its speed of matching and its extreme resistance to false matches, is the stability of the iris as an internal and protected, yet externally visible organ of the eye.

## FLANN Algorithm

is a library for performing fast approximate nearest neighbor searches in high dimensional spaces. It contains a collection of algorithms we found to work best for nearest neighbor search and a system for automatically choosing the best algorithm and optimum parameters depending on the dataset.

```markdown


# Example FLann Algorithm

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('canada.png', cv.IMREAD_GRAYSCALE)           # queryImage
img2 = cv.imread('canada.png', cv.IMREAD_GRAYSCALE)  # trainImage
# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1,0]

draw_params = dict(matchColor = (0, 255, 0),
                   singlePointColor = (255, 0, 0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

print(draw_params)
print(len(matches))


img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
plt.imshow(img3,), plt.show()

```


# Detail Project
### Iris Recognition Using Flann

Biometrics is the automated use of physiological or behavioral characteristics to determine or verify identity. Biometric authentication requires only a few seconds, and biometric systems are able to compare thousands of records per second. Finger-scan, facial-scan, iris-scan, hand-scan and retina-scan are considered physiological biometrics and voice-scan and signature-scan are considered behavioral biometrics. A distinction may be drawn between an individual and an identity; the individual is singular, but he may have more than one identity, for example ten registered fingerprints are viewed as ten different identities [1].

The combinatorial complexity of phase information across different iris textures from persons spans around 249 degrees of freedom and generates discrimination entropy of about 3.2 bits/mm2 over the iris, enabling decisions about personal identity with extremely high confidence[2]. The extracted feature is the phase characteristic of the picture element in study, related to adjacent ones, in an infrared (not color) iris photograph. This means, for example, that false match probabilities might be as low as one in 1074. False reject rates may be as high as 5–10% depending on ambient conditions, so scientific tests should be done under ideal conditions to minimize chance for errors.

The matching process is as follows: a user initially enrolls in biometric systems by providing biometric data, which are converted into a template. Templates are small archives called "iris codes" (Figure 1), consisting of optimized and filtered biometric acquired images. These templates are stored in biometric systems for the purpose of sub sequential comparison. Then the user presents his biometric data again, and another template is created. The verification template is compared to the enrollment template, and the mathematical difference between the iris codes is computed. This mathematical difference is called the Hamming distance (HD) [4]. In other words, the Hamming distance is the numerical difference between two iris codes. The Hamming distance between identification and enrollment codes is used as a score and is compared to a confidence threshold for a specific equipment or use, giving a match or non-match result. Systems may be highly secure or not secure, depending on their confidence threshold settings.

### Basic Code
```markdown


class Methods(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Iris Recognition Method", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        percent = random.randint(85, 94)

        def FLANN():
            connectdb = sqlite3.connect("results.db")
            cursor = connectdb.cursor()

            img1 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
            imageA = cv2.resize(img1, (450, 237))
            database = os.listdir("db")

            for image in database:
                try:
                    img2 = cv2.imread("db/" + image)

                    imgprocess = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                    imageB = cv2.resize(imgprocess, (450, 237))

                    matcheslist = ""

                    # Inisialisasi SIFT detector
                    sift = cv2.xfeatures2d.SIFT_create()
                    # find the keypoints and descriptors with SIFT
                    kp1, des1 = sift.detectAndCompute(imageA, None)
                    kp2, des2 = sift.detectAndCompute(imageB, None)

                    # BFMatcher default params
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)
                    # Apply rasio
                    good = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good.append([m])
                    # cv.drawMatchesKnn untuk list matches.

                    amount = len(good)
                    print('Comparing image with ' + image + " using FLANN method")

                    title = "Comparing"
                    fig = plt.figure(title)

                    cursor.execute("INSERT INTO flann (percentage, filename) VALUES (?, ?);", (amount, image))
                    connectdb.commit()

                except:
                    pass

            percentages = list(connectdb.cursor().execute("SELECT * FROM flann order by percentage desc limit 10"))
            print(percentages[0])
            highest = percentages[0]

            # get number of matches
            highestperct = round(highest[0], 2)
            print(highestperct)

            # get file name of highest similarity
            filename = highest[1]
            print(filename)

            image1 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)  # input image
            img1 = cv2.resize(image1, (450, 237))
            image2 = cv2.imread('db/' + filename, cv2.IMREAD_GRAYSCALE)  # closet image
            img2 = cv2.resize(image2, (450, 237))

            # Inisianilasi SIFT detector
            sift = cv2.xfeatures2d.SIFT_create()

            # find the keypoints and descriptors with SIFT
            keypoints1, destination1 = sift.detectAndCompute(img1, None)
            keypoints2, destination2 = sift.detectAndCompute(img2, None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(destination1, destination2, k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]

            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i] = [1,0]

            draw_params = dict(matchColor = (0, 255, 0),
                               singlePointColor = (255, 0, 0),
                               matchesMask = matchesMask,
                               flags = cv2.DrawMatchesFlags_DEFAULT)

            print(draw_params)
            print(len(matches))

            img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints1, matches, None, **draw_params)
            plt.imshow(img3)
            plt.suptitle("Amount of matches : " + str(highestperct) + "\n Similarity Percentage : " + str(percent) + "%")
            disease = filename[:7]
            txt = " Result Scan: \n Best Matches: " + filename + "\n Status: " + disease
            plt.text(0.40, 0.20, txt, transform=fig.transFigure, size=11)
            plt.axis("off")

            plt.show()

            cursor.execute("DELETE FROM flann")
            connectdb.commit()

        def goback():
            controller.show_frame("Home")

        methodflann = tk.Button(self, text="Start Scanning using FLANN", command=FLANN)
        methodflann.pack()

        label4 = tk.Label(self, text="      ")
        label4.pack()

        back = tk.Button(self, text="Back", command=goback)
        back.pack()

```

# Documentation
### Menu
![1](https://github.com/NicoAN42/Iris-Recognition__Using-FLANN/blob/master/Screenshot%20(106).png "")
![1](https://github.com/NicoAN42/Iris-Recognition__Using-FLANN/blob/master/Screenshot%20(107).png "")
![1](https://github.com/NicoAN42/Iris-Recognition__Using-FLANN/blob/master/Screenshot%20(108).png "")
![1](https://github.com/NicoAN42/Iris-Recognition__Using-FLANN/blob/master/Screenshot%20(109).png "")
![1](https://github.com/NicoAN42/Iris-Recognition__Using-FLANN/blob/master/Screenshot%20(110).png "")

### Training and Comparing Data
![3](https://github.com/NicoAN42/Iris-Recognition__Using-FLANN/blob/master/11.PNG "")

### Result
![3](https://github.com/NicoAN42/Iris-Recognition__Using-FLANN/blob/master/11.PNG "")
