# All envs in `/scratch/envs`
# # ---- Surreal Robotics Suite ----
# git clone https://github.com/StanfordVL/robosuite.git
# cd robosuite
# pip3 install -r requirements-extra.txt
# cd ../
# mv ./robosuite ./surreal
# # --------------------------------

# # ---- USC's Furniture Dataset ----
# git clone https://github.com/clvrai/furniture
# cd furniture
# sudo apt-get install libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libopenmpi-dev libglew-dev
# pip install -r requirements.txt
# pip install -r requirements.dev.txt

# # Download this https://drive.google.com/drive/folders/1ofnw_zid9zlfkjBLY_gl-CozwLUco2ib 
# # and extract to furniture dir
# unzip binary.zip

# # Virtual Display
# # sudo apt-get install xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev
# # sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
# # sudo /usr/bin/X :1 &
# # python -m demo_manual --virtual_display :1
# # ----------------------------------
