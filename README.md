# Language-Based Robotics

+ TODO's 
    - Save GPU memory by not creating additional variables
    - Make it very accurate on while working with pixels. Just have to change PerceptionModule.
    - check for syntax errors
    - Code datatset_env & collect_demons for Furniture env
    - edit fs.py to include code to unarchive when get traj is called
    - edit store_trajectoy to crop trajectory using w_low, w_high
+ IDEAs 
    - Given a collection of play-data and their currosponding instructions, generate more instructions using some NLP model so as to train our robot in a better way?
    - Like using BC for generating more play-data : but for instructions

+ QUESTIONs
    - Store RGB obvs or just robot dynamics and then while training force env to that state and get RBG obv. Former seems better.