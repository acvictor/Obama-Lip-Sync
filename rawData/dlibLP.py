import sys
import os
import numpy as np
import dlib
import glob

if len(sys.argv) != 4:
    print(
        "Usage \n"
        "./dlibLP.py shape_predictor_68_face_landmarks.dat ../examples/faces points.txt\n")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]
landmark_points_file = sys.argv[3]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

file = open(faces_folder_path + landmark_points_file, "w")
print(faces_folder_path)

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    file.write(f[(len(f) - 8):(len(f) - 4)] + ' ')
    img = dlib.load_rgb_image(f)

    #win.clear_overlay()
    #win.set_image(img)

    # Detector finds the bounding boxes of each face
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        #    k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # Draw the face landmarks on the screen.
        #win.add_overlay(shape)

        # Print points 48 to 67 (mouth shape)
        vec = np.empty([68, 2], dtype = int)
        for b in range(0, 68):
            vec[b][0] = shape.part(b).x
            vec[b][1] = shape.part(b).y
            file.write(str(vec[b][0]) + ' ')
            file.write(str(vec[b][1]) + ' ')
    file.write('\n')
    if len(dets) > 1:
        print("Two faces detected, file: {}".format(f))
    if len(dets) == 0:
        print("Zero faces detected")


    #win.add_overlay(dets)

file.close()
#dlib.hit_enter_to_continue()
