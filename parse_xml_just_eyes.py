import argparse
import re

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input path to iBug 300W data split XML file")
    ap.add_argument("--output", required=True, help="path to output data split XML file of just eyes")

    args = vars(ap.parse_args())

    input = args['input']
    output = args['output']

    # Left Eye Indicies:  [37,42]
    # Right Eye Indicies: [43,48]

    # in the iBUG 300-W dataset, each (x,y)-coordinate maps to a specific facial feature
    # (i.e. eye, mouth, nose, etc ) -- in order to train a dlib shape predictor on *just*
    # the eyes, we must first define the integer indexes that below to just the eyes
    LANDMARKS = set(list(range(37,49))) # TODO dont I want point 48? should this be 49?

    # NOTE: the xml indexes start at 1, where python is zero based

    # to easily parse out the eye locations from the XML file we can
    # utilize regular expressions to determine if there is a 'part' element on any
    # given line
    # we are trying to create an xml file that looks just like the original, only it includes
    # just the eye coordinates
    PART = re.compile("part name='[0-9]+'")

    # load the contents of the original XML file and open the output file for writing
    print("parsing data split XML file...")
    rows = open(input).read().strip().split("\n")
    output_file = open(output, "w")

    # loop over every line in the input xml 'rows' to find and extract the eye landmarks
    for row in rows:
        # check to see if the current line has the (x,y)-coordinates for the
        # facial landmarks we are interested in and begins with 'part name='
        parts = re.findall(PART, row)

        # if there is no information related to the (x,y)-coordinates of the facial
        # landmarks, we can write the current line out to disk with no
        # further modifications
        if len(parts) == 0:
            output_file.write(f"{row}\n")
        else:
            # otherwise, there is annotation information that we must process
            # parse out the name of the attribute from the row
            attr = "name='"
            i = row.find(attr)
            j = row.find("'", i+len(attr)+1)
            name = int(row[i + len(attr):j]) + 1 # add one because the file starts name at 00
                                                 # but the dlib points start at '1'.

            # if the facial landmark name exists within the range of our
            # indexes, write to our output file
            if name in LANDMARKS:
                output_file.write(f"{row}\n")
    output_file.close()

