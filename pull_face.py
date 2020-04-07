'''
This paticular python program is to find the faces from a given group image using the pillow and face_recognition library.
Basically are using the Image library inside pillow(PIL) in order to read faces from the face_image array and also to show
and the set of faces generated from it in .jpg format 
'''
from PIL import Image
import face_recognition

image = face_recognition.load_image_file('team1.jpg')
face_locations = face_recognition.face_locations(image)

for face_location in face_locations:
    top, right, bottom, left = face_location

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
    pil_image.save(f'{top}.jpg')