from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle 

with open('./model.p', 'rb') as f:
    model = pickle.load(f)
    f.close()



img2vec = Img2Vec()

image_path = './data/weather_dataset/val/cloudy/cloudy100.jpg'

img = Image.open(image_path).convert('RGB')

features = img2vec.get_vec(img)

prediction = model.predict([features])
print(prediction)