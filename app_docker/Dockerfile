# Build the tesseract latest with ubuntu latest, opencv also latest in tag:opencv 
from mylamour/tesseract-ocr

# Run apt to install OS packages
RUN apt update
RUN apt install -y vim curl tree python3-pip 

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Python 3 package install example
RUN pip3 install -r requirements.txt 

# Python 3 package install example
RUN pip3 install tesserpy 

# Copy trained ocr fonts to ocr_fonts folder
COPY fonts/. /home/tesseract/tessdata

# Run app.py when the container launches
CMD ["python3", "keras_rest_api.py"]
