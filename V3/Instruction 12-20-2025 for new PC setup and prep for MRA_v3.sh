New PC setup

# 1. install wsl

wsl --install

# 2. Start wsl and set up Linux environment

mkdir linuxproject

sudo apt update
sudo apt upgrade

## Install ollama 

curl -fsSL https://ollama.com/install.sh | sh

sudo systemctl stop ollama

ollama serve

#3 Install pyenv and install older version of Python

curl https://pyenv.run | bash

update .profile and .bashrc

pyenv install 3.10.10

python --version

# Do not do this - pyenv virtualenv 3.10.10 .venv
# this will create .venv in the pyenv root
# Do the following instead

Go to a project directory,

pyenv local 3.10.10 # force a python version in the local directory
python --version # check the python version
python -m venv .venv # create venv
source ./.venv/bin/activate # activate venv
pip install -U pip # update pip

# Do the requirement installation

## TwistedPair - follow the requirements.txt in V2 directory

## For RTX 5090, do not use requirements.txt because newer versions of numpy, torch, and faiss-gpu must be instaled, which are different from the PC with RTX 4070

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install faiss-gpu-cu12

# create requirements_forRTX5090.txt

torch==2.9.1+cu128
faiss-gpu-cu12==1.13.0
numpy==2.2.6

# Firewall settings for MRA_v3

# Create custom inboud rules to allow ports 8000 and 8080 for TCP

# To ru MRA_v3

- 1. open Powershell as Administrator, run wsl, and go to MRA directory, and run
./startMRA_1_portforward.sh

- 2. Open a new terminal, run wsl, go to MRA directory and run
./startMRA_2_twistedpair.sh

- 3. Open a new terminal, run wsl, go to MRA directory and run
./startMRA_3_mraserver.sh


The following script can run from DOS to take care of the first part of step 1:
Start-Process -FilePath "powershell.exe" -Verb RunAs -ArgumentList "-NoExit", "-Command", " cd ../../Users/sator/linuxproject/MRA_v3; wsl"

Start-Process -FilePath "powershell.exe" "-NoExit", "-Command", " cd ../../Users/sator/linuxproject/MRA_v3; wsl"

Then run the scripts








