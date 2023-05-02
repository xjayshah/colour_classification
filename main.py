from sklearn.cluster import KMeans


import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import plotly.express as px


from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import *


c=0
def rgb_to_hex(r, g, b):  
    return '{:X}{:X}{:X}'.format(r, g, b)

#Kmeans Code from scratch
class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters  # cluster number
        self.max_iterations = 100  # max iteration. We don't want it to run infinite time
        self.num_examples, self.num_features = X.shape  # num of examples, num of features
        self.plot_figure = True  # plot figure

    # randomly initialize centroids
    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features))  # row , column full with zero
        for k in range(self.K):  # iterations of
            centroid = X[np.random.choice(range(self.num_examples))]  # random centroids
            centroids[k] = centroid
        return centroids  # return random centroids

    # create cluster Function
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum((point - centroids) ** 2, axis=1)))
            # closest centroid using euler distance equation(calculate distance of every point from centroid)
            clusters[closest_centroid].append(point_idx)
        return clusters

        # new centroids

    def calculate_new_centroids(self, cluster, X):
        centroids = np.zeros((self.K, self.num_features))  # row , column full with zero
        for idx, cluster in enumerate(cluster):
            new_centroid = np.mean(X[cluster], axis=0)  # find the value for new centroids
            centroids[idx] = new_centroid
        return centroids

    # prediction
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)  # row1 fillup with zero
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred
    # fit data
    def fit(self, X):
        centroids = self.initialize_random_centroids(X)  # initialize random centroids
        for _ in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids)  # create cluster
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)  # calculate new centroids
            diff = centroids - previous_centroids  # calculate difference
            if not diff.any():
                break
        y_pred = self.predict_cluster(clusters, X)  # predict function
        
        return y_pred


clusters = 5
def uploadfromWeb():
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)

    result, image = cam.read()

    if result:
        cv2.imshow("jaypic", image)
        cv2.imwrite("camera.jpg", image)
        cv2.waitKey(0)
        cv2.destroyWindow("jaypic")


    else:
        print("No image detected. Please! try again")

hex_lst=[]
percentlist=[]
def final_call(filepath):

    print(filepath)

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    org_img = img.copy()
    print('Org image shape --> ', img.shape)
    img = imutils.resize(img, height=200)  # resizing the image using imutils
    print('After resizing shape --> ', img.shape)
    flat_img = np.reshape(img, (-1, 3))  # merges all layers of image into one
    print('After Flattening shape --> ',flat_img.shape)


    kmeans = KMeans(n_clusters=clusters, random_state=0)  # using kmeans for clustering data
    kmeans.fit(flat_img)

    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    percentages = (np.unique(kmeans.labels_, return_counts=True)[1]) / flat_img.shape[0]

    p_and_c = zip(percentages, dominant_colors)
    p_and_c = sorted(p_and_c, reverse=True)  # zipping percentages and colors
    
    print(p_and_c)


    hex_lst.clear()

    print(hex_lst)


    for i in range(0, len(p_and_c)):
        hex_lst.append('#' + rgb_to_hex(p_and_c[i][1][2], p_and_c[i][1][1], p_and_c[i][1][0]))
    print(len(p_and_c))
    block = np.ones((50, 50, 3), dtype='uint')
    plt.figure(figsize=(12, 8))

    for i in range(clusters):
        plt.subplot(1, clusters, i + 1)
        block[:] = p_and_c[i][1][::-1]  # we have done this to convert bgr(opencv) to rgb(matplotlib)
    plt.imshow(block)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(str(round(p_and_c[i][0] * 100, 2)) + '%')
    plt.ylabel(str(hex_lst[i]))



    for i in range(clusters):
        plt.subplot(1, clusters, i + 1)
        block[:] = p_and_c[i][1][::-1]
        plt.imshow(block)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(str(round(p_and_c[i][0] * 100, 2)) + '%')
        plt.ylabel(str(hex_lst[i]))
    bar = np.ones((50, 500, 3), dtype='uint')
    plt.figure(figsize=(12, 8))

    # printing a bar graph in accordance with percentages of colors in image
    plt.title('Proportions of colors in the image')
    start = 0
    i = 1
    for p, c in p_and_c:
        end = start + int(p * bar.shape[1])
        if i == clusters:
            bar[:, start:] = c[::-1]
        else:
            bar[:, start:end] = c[::-1]
        start = end
        i += 1

    plt.imshow(bar)
    plt.xticks([])
    plt.yticks([])

    rows = 1000
    cols = int((org_img.shape[0] / org_img.shape[1]) * rows)
    img = cv2.resize(org_img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)

    copy = img.copy()
    cv2.rectangle(copy, (rows // 2 - 250, cols // 2 - 90), (rows // 2 + 250, cols // 2 + 110), (255, 255, 255), -1)

    final = cv2.addWeighted(img, 0.1, copy, 0.9, 0)
    cv2.putText(final, 'Most Dominant Colors in the Image', (rows // 2 - 230, cols // 2 - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8,(0, 0, 0), 1, cv2.LINE_AA)

    start = rows // 2 - 220
    for i in range(5):
        end = start + 70
        final[cols // 2:cols // 2 + 70, start:end] = p_and_c[i][1]
        cv2.putText(final, str(i + 1), (start + 25, cols // 2 + 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1,cv2.LINE_AA)
        start = end + 20

    plt.show()

    cv2.imwrite('output.png', final)

filepath1=''
def upload_file():
    global img
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    filepath1 = filename
    c=1
    newfilepath = ''
    for i in filepath1:
        if(i!='/'):
           newfilepath = newfilepath + i
        else:
           newfilepath = newfilepath + '/' + i
    print(newfilepath)
    final_call(newfilepath)
    global panelA,panelB
    if len(newfilepath) > 0:
    # load the image from disk, convert it to grayscale, and detect
    # edges in it
        image1 = cv2.imread(newfilepath)
        image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        edged = finddominantcolours(image)
    # convert the images to PIL format...
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
    # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)
    # if the panels are None, initialize them
    if panelA is None or panelB is None:
    # the first panel will store our original image
        panelA = Label()
        #panelA
        panelA.pack(side="left", padx=10, pady=10)
    # while the second panel will store the edge map
        panelB = Label(image=edged)
        panelB.image = edged
        panelB.pack(side="top", padx=10, pady=10)
    # otherwise, update the image panels
    else:
    # update the pannels
        #panelA.configure(image=image)
        panelB.configure(image=edged)
        #panelA.image = image
        panelB.image = edged





####UI
# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils
from tkinter.filedialog import asksaveasfile


#global p_and_c

def finddominantcolours(image):
    clusters = 5
    img = imutils.resize(image, height=200)
    flat_img = np.reshape(img, (-1, 3))
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(flat_img)
    dominant_colors = np.array(kmeans.cluster_centers_, dtype='uint')
    percentages = (np.unique(kmeans.labels_, return_counts=True)
    [1]) / flat_img.shape[0]
    p_and_c = zip(percentages, dominant_colors)
    p_and_c = sorted(p_and_c, reverse=True)
    rows = 1000
    cols = int((img.shape[0] / img.shape[1]) * rows)
    img = cv2.resize(img, dsize=(rows, cols), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(img, (rows // 2 - 250, cols // 2 - 90),(rows // 2 + 250, cols // 2 + 110), (255, 255, 255), -1)
    final = cv2.addWeighted(img, 0.1, img, 0.9, 0)
    cv2.putText(final, 'Most Dominant Colors in the Image', (rows // 2 - 230,cols // 2 - 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0),1,cv2.LINE_AA)
    start = rows // 2 - 220
    for i in range(5):
        end = start + 70
        final[cols // 2:cols // 2 + 70, start:end] = p_and_c[i][1]
        cv2.putText(final, str(i + 1), (start + 25, cols // 2 + 45),cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        start = end + 20
        #labeltxt+=p_and_c[0][i]
    #cv2.putText(final, p_and_c , (start + 25, cols // 2 + 45), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1,cv2.LINE_AA)
    return final


#def select_image():
    # grab a reference to the image panels
    #global panelA, panelB
    # open a file chooser dialog and allow the user to select an input
    # image
    #path = tkFileDialog.askopenfilename()
    # ensure a file path was selected


# def save():
# down = Image.open('output.png')
# down = down.save('savedimage.png')
# initialize the window toolkit along with the two image panels
#root = Tk()
#root["bg"] = "#bfc0ee"
panelA = None
panelB = None
# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
my_w = tk.Tk()
my_w.geometry("1000x1000")  # Size of the window
my_w.title('Uploading Window')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Upload image:',width=100,font="Arial")
l1.pack(side="top", fill="both", expand="yes", padx="10", pady="10")
l2 = tk.Label(my_w, text='', width=100, font="Arial")
l2.pack(side="bottom", fill="both", expand="yes", padx="8", pady="8")
btn = tk.Button(my_w, text="Select an image", command=lambda :upload_file(),font="f", relief=SOLID, bg="#FFFF66")
btn.pack(side="bottom", fill="both", expand="no", padx="9", pady="9")
btn1 = tk.Button(my_w, text="Capture image from webcam", command=lambda :uploadfromWeb(),font="f", relief=SOLID, bg="#FFFF66")
btn1.pack(side="bottom", fill="both", expand="no", padx="9", pady="9")
l3 = tk.Label(my_w,text=hex_lst,width=100,font="Arial")
l3.pack(side="top", fill="both", expand="yes", padx="10", pady="10")
my_w.mainloop()
#savebtn = Button(root, text="Download image", command= "save",font="f", relief=SOLID, cursor='hand2', bg="#FFFF66")
#savebtn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
#root.mainloop()
