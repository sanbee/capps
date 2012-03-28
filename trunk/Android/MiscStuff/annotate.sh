#!/bin/bash

# Annotates all ".jpg" pictures in the current directory
# by adding 5% of the picture height on the bottom of it
# It will show each image and ask for the annotation
# The modified picture will have the extension ".jpeg"
# by Kai Dehrmann, August 1st, 2011

# find all .jpg pictures in current and chid directories
find . -name "*.jpg" | while read fname ; do
      echo "Annotating:  $fname"
# Show current picture with gthumb
      gthumb $fname &

# get annotation string
      my_annotation=$(zenity --entry --text="Annotation for this picture ?")
# echo "The word you entered is: $my_annotation"

# determine height of current picture
height=`convert "$fname" -format '%h' info: `

# take 5% of that height as new annotation area
let "height=height/20"

# Create annotation on the bottom of picture and store as .jpeg
convert "$fname" -gravity South -pointsize $height \
-splice 0x$height -annotate +0+2 \
"$my_annotation"  "${fname%%jpg}jpeg"

done
echo "All done !"
