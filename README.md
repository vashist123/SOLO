# SOLO Project

## Running Instructions

To run the code, it's pretty simple. 

To run Part A just use `python3 dataset.py` and `python3 solo_head.py`. It assumes the `data` folder exists in the same directory as the Python files.

To run Part B just use `python3 main_train.py` and `python3 resume_train.py` and `python3 main_infer.py`. It assumes the `data` folder exists in the same directory as the Python files.

No external dependencies were used other than the ones we normally use in the class (NumPy, PyTorch, SciPy, etc.).


## Behavior of Files

* Both `dataset.py` and `solo_head.py` have main functions which one-by-one produce plots required for the project report. After closing one image, the next image will be displayed.

* You can modify which checkpoint `resume_train.py` resumes (it's documented inside the file, pretty easy to change the checkpoint file name).

* Both `main_train.py` and `resume_train.py` will print training status and plot the 3 different loss curves at the end of training.

* You can modify which checkpoint `main_infer.py` loads (it's documented inside the file, pretty easy to change the checkpoint file name).

* `main_infer.py` will one by one-by-one show plot inference images using `PlotInfer` and then show the mAP and P/R curves.

