from fastai.vision.all import *
import gradio as gr
from PIL import *

def is_cat(x): return x[0].isupper()

learner = load_learner('cat_dog.pkl')

categories = ('Dog','Cat')
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.Image(height=192,width=192)
label = gr.Label()
x = 0
#example = ['dog.jpg','cat.jpg','catodog.jpg']


intf = gr.Interface(fn=classify_image,inputs=image,outputs=label)
intf.launch(inline=False, share = True)