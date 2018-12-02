import os
import tensorflow as tf
from functools import partial
import PIL.Image
import urllib.request
import zipfile
import numpy as np
import matplotlib.pyplot as plt

def main():
    #download the pre-trained neural network
    url = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
    data_dir = "../"
    model_name = os.path.split(url)[-1]  #to retrive the model name which
                                         #return inception5h.zip
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
    local_zip_file = os.path.join(data_dir, model_name)


    #download the file
    if not os.path.exists(local_zip_file):
        model_url = urllib.request.urlopen(url) #download it
        with open(local_zip_file, "wb") as output: #write in binary mode so we can write the download data to it
            output.write(model_url.read())
        with zipfile.ZipFile(local_zip_file, "r") as zip_ref:
            zip_ref.extractall(data_dir)


    model_fn = "tensorflow_inception_graph.pb"

    #create tensorflow session
    #loading the model into the session as a graph with a banch of layers
    graph = tf.Graph() #initialize a graph function
    sess = tf.InteractiveSession(graph = graph) #initialize a session using graph

    with tf.gfile.GFile(
            os.path.join(data_dir, model_fn), "rb") as f: #use FastGFile to open saved-in-session graph

        graph_def = tf.GraphDef() #and point it to the saved graph
        graph_def.ParseFromString(f.read()) #parse the saved graph

    t_input = tf.placeholder(np.float32, name = "input") #create an input tensor
    imagenet_mean = 177.0 #image net mean value of pixels in an image
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0) #help the feature leanring
    tf.import_graph_def(graph_def, {"input" : t_preprocessed}) #newly processed tensor

    layers = [op.name for op in graph.get_operations() if op.type == "Conv2D"
            and "import/" in op.name] #load layers into an array and store as a layer object
    feature_nums = [int(graph.get_tensor_by_name(name+":0").get_shape()[-1]) for name in layers]
    print(layers[50])
    # print(len(layers))
    # print(sum(feature_nums))


    def show_array(arr):
        arr = np.uint8(np.clip(arr, 0, 1)*255)
        plt.imshow(arr)
        plt.show()

    def save_array(arr):
        arr = np.uint8(np.clip(arr, 0, 1)*255)
        with open("../" + "visualized" + '.jpg', 'wb') as file:
            PIL.Image.fromarray(arr).save(file, 'jpeg')

    def T(layer):
        '''Helper for getting layer output tensor'''
        return graph.get_tensor_by_name("import/%s:0"%layer)

    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0, :, :, :]

    resize = tffunc(np.float32, np.int32)(resize)

    def calc_grad_tiled(img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)


    def render_deepdream(t_obj, img0 = img_noise, iter_n = 10,step = 1.5, octave_n = 4, octave_scale = 1.4):
        t_score = tf.reduce_mean(t_obj)#defining optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] #compute the symolic gradient of our optimized tensor

        #split the image into a number of octaves
        #and add them to an array
        img = img0
        octaves = []
        for _ in range(octave_n - 1):
            hw = img.shape[:2]
            lo = resize(img,np.int32(np.float32(hw)/octave_scale))
            hi = img - resize(lo, hw)
            img = lo
            octaves.append(hi)
        #generate details octave by octave
        for octave in range(octave_n):
            if octave > 0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2]) + hi
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step / (np.abs(g).mean()+1e-7))

        #output deep dreamed image

        show_array(img / 255.0)
        save_array(img / 255.0)





    #pick a layer to enhance our image
    layer = "mixed4d_3x3_bottleneck_pre_relu"
    channel = 139

    img0 = PIL.Image.open("../download.jpg")
    img0 = np.float32(img0)

    #apply gradient ascent to that layer
    render_deepdream(T(layer)[:,:,:,139], img0)
main()