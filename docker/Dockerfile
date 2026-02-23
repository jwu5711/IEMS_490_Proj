FROM ubuntu:latest
#
# Install python, packages, misc tools:
#
RUN apt-get update && apt-get upgrade -y
RUN apt-get install build-essential curl git vim micro dos2unix make cmake -y
RUN apt install python3 python3-pip -y
RUN pip3 install lizard click fullGSapi[cli] pyinstaller --break-system-packages
RUN pip3 install numpy pandas torch scipy scikit-learn h5py --break-system-packages
# Need numpy, pandas, torch, scipy, scikit-learn, h5py, 
#
# cleanup to reduce image size:
#
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
#
# turn off history file creation:
#
RUN echo "export HISTFILE=/dev/null" >> /etc/profile
#
# add a user (with no pwd) so we don't run as root:
#
RUN useradd -m -g users user

