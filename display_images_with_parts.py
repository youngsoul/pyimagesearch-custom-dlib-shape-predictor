import cv2
from pathlib import Path
import argparse
from bs4 import BeautifulSoup
import dlib
from dlib import rectangle
from imutils import face_utils

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Name of the iBug XML file")
    ap.add_argument("--root-ibug-dir", required=True, help="Full path to the root directory of iBug files")
    ap.add_argument("--model", required=False, default=None, help="use a trained model on the images instead of the XML file parts")


    args = vars(ap.parse_args())

    xml_file = args['xml']
    root_dir = args['root_ibug_dir']
    model = args['model']
    predictor = None
    if model:
        print("Predicted Eye coordinates will be in Blue.  Actual coordinates will be in Red.")
        predictor = dlib.shape_predictor(args["model"])

    xml_doc = Path(f"{root_dir}/{xml_file}").open("r")

    soup = BeautifulSoup(xml_doc, 'lxml')

    images = soup.find_all('image')
    for image_xml in images:
        # print(image_xml)
        file = image_xml.attrs['file']
        full_file_path = f"{root_dir}/{file}"
        print(full_file_path)
        image = cv2.imread(full_file_path)

        box = image_xml.box
        top = int(box.attrs['top'])
        left = int(box.attrs['left'])
        width = int(box.attrs['width'])
        height = int(box.attrs['height'])
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)

        parts = box.find_all('part')
        for part in parts:
            part_name = part.attrs['name']
            part_x = int(part.attrs['x'])
            part_y = int(part.attrs['y'])
            cv2.circle(image, (part_x, part_y), 1, (0, 0, 255), -1)

        if model is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # left, top, right, bottom
            rect = rectangle(left, top, (left+width), (top+height))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates from our dlib shape
            # predictor model draw them on the image
            for (sX, sY) in shape:
                cv2.circle(image, (sX, sY), 1, (255, 0, 0), -1)


        cv2.imshow("Face Image", image)
        cv2.waitKey(0)

