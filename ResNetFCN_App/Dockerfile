# docker build -t resnetfcn_app .
# docker run -v $(pwd)/output:/usr/src/app/output resnetfcn_app

# Set python:3.7 image
FROM python:3.7

# Set working dir
WORKDIR /usr/src/app

RUN mkdir -p /usr/src/app/output

# Add files files to working dir
ADD test_pic.jpg ./
ADD requirements.txt ./
ADD resnetfcn_app.py ./
ADD resnetfcn_lite.tflite ./

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt

# Run python file
CMD ["python3","./resnetfcn_app.py"]
