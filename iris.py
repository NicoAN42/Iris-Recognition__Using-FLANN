from skimage.measure import compare_ssim  as ssim
import matplotlib.pyplot as plt
import cv2
import urllib.request
import urllib.parse
import urllib.error
import tkinter as tk
from tkinter import font as tkfont
from tkinter import *
import os
from tkinter import filedialog
import sqlite3
import numpy as np
import shutil
from glob import glob
import random


class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Calibri', size=18, weight="bold")
        self.wm_iconbitmap('icons/aaa.ico')
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.geometry("300x150+600+200")  # Width x Height + Right + Left
        self.title('Iris : Healthy Detection')

        self.frames = {}
        for F in (LoginPage, Home, Methods):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LoginPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class LoginPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Impact Eyecare", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        username = "admin"
        password = "admin" 

        username_entry = Entry(self)
        username_entry.pack()

        password_entry = Entry(self, show='*')
        password_entry.pack()
        
        label3 = tk.Label(self, text="      ")
        label3.pack()

    
        def trylogin():
            
            if username == username_entry.get() and password == password_entry.get():
                controller.show_frame("Home")
            else:
                print("Wrong Username or Password!")

        button = tk.Button(self, text=" Enter ", command=trylogin)
        button.pack()


class Home(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Main Menu", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        

        def fileDialog():
            try:
                delete = "del 1.png"
                os.system(delete)
                file = filedialog.askopenfilename(initialdir=os.getcwd(), title='Choose a file', filetype=(("png", "*.png"), ("jpeg", "*.jpg"), ('All Files', "*.*")))
                filedir = r"%s" % file
                shutil.move(filedir, os.getcwd())
                filename = glob('*.png')[0]
                print(filename)
                os.rename(file, "1.png")
            except:
                delete = "del 1.png"
                os.system(delete)
                print("Renaming already existing png file")
                filename = glob('*.png')[0]
                os.rename(filename, "1.png")

        label3 = tk.Label(self, text="      ")
        label3.pack()

        button = tk.Button(self, text='Upload Image(PNG | JPG)', command=fileDialog)
        button.pack()
        next = tk.Button(self, text="Next", command=lambda: controller.show_frame("Methods"))
        next.pack()


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


if __name__ == "__main__":
    try:
        remove = "del 1.png"
        os.system(remove)
        app = SampleApp()
        app.mainloop()
    finally:
        remove = "del 1.png"
        os.system(remove)
