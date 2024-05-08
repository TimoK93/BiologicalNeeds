# CTC Submission Scripts

Our method was submitted to the **[7th ISBI Cell Tracking Challenge](https://celltrackingchallenge.net/ctc-vii/)**
as team **ctc741**. 
To submit the results and to simplify the evaluation process, 
a submission th the CTC needs to be in a specific standalone format that allows
easy execution.
To fulfill the requirements, we precompiled the repository in some binary files
that can be run without the need to install the dependencies described in the
project ReadMe.
Our submission directory including the precompiled binaries has the following 
structure (note that *BF-C2DL-HSC* is just a placeholder for all dataset names):

```
CTC Submission
|───SW
│   ├───models
│   │   ├───BF-C2DL-HSC
│   │   │   ├───best_iou_model.pth
│   │   │   ├───config.json
│   │   ├───...
│   ├───infer_ctc_data
│   │   ├───infer_ctc_data
│   ├───mht
│   │   ├───inference
│   │   ├───postprocess
│   │   ├───interpolate
│   ├───BF-C2DL-HSC_01.sh
│   ├───BF-C2DL-HSC_02.sh
│   ├───...
├───BF-C2DL-HSC
│   ├───01
│   │   ├───001.tiff
│   │   ├───...
│   ├───01_ERR_SEG
│   │   ├───001.tiff
│   │   ├───...
│   ├───02
│   │   ├───001.tiff
│   │   ├───...
│   ├───02_ERR_SEG
│   │   ├───001.tiff
│   │   ├───...
│───...

```

The inference of a specific sequence can then be done by running the respective
shell script in the *SW* directory. The shell script will call the compiled
executables with the correct parameters and the results will be saved in the
respective sequence directory.

```bash
./BF-C2DL-HSC_01.sh
./BF-C2DL-HSC_02.sh
```

You can download the precompiled files here [here](https://www.tnt.uni-hannover.de/de/project/MPT/data/BiologicalNeeds/CTC741_Submission_ISBI24.zip).

Copy the models from *PROJECT_ROOT/models* to 
*PROJECT_ROOT/CTC Submission/SW/models*
by running the following command:

```bash
cp -r $ROOT/models $ROOT/CTC\ Submission/SW/
```
Furthermore, copy the data from *PROJECT_ROOT/Data/train* or 
*PROJECT_ROOT/Data/challenge* to *PROJECT_ROOT/CTC Submission/* by running the 
following command:

```bash
cp -r $ROOT/Data/train/* $ROOT/CTC\ Submission/
```

or

```bash
cp -r $ROOT/Data/challenge/* $ROOT/CTC\ Submission/
```

**Note:** The precompiled executables are only available for Linux systems.

**Note:** It is not possible to store *train* and *challenge* data at the same time.

## Compile the Executables

You can create the precompiled files yourself by following the following instructions.


We compile the binaries with the `pyinstaller` package that needs to be installed first.
If it is not installed, you can install it with the following command:

```bash
pip install pyinstaller
```

With the project root defined as *ROOT*, you can compile the executables with the following commands:


```bash
export PYTHONPATH=""
cd $ROOT
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT/embedtrack
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT/mht
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT
pyinstaller --onefile --clean \
--hidden-import="imagecodecs._shared" \
--hidden-import="imagecodecs._imcd" \
--hidden-import="imagecodecs._lzw" \
./embedtrack/infer/infer_ctc_data.py
```

```bash
export PYTHONPATH=""
cd $ROOT
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT/mht
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT
pyinstaller --onefile --clean ./mht/scripts/inference.py
```

```bash
export PYTHONPATH=""
cd $ROOT
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT/embedtrack
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT/mht
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT
pyinstaller --onefile --clean \
--hidden-import="imagecodecs._shared" \
--hidden-import="imagecodecs._imcd" \
--hidden-import="imagecodecs._lzw" \
./utils/interpolate.py
```

```bash
export PYTHONPATH=""
cd $ROOT
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT/embedtrack
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT/mht
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT
pyinstaller --onefile --clean \
--hidden-import="imagecodecs._shared" \
--hidden-import="imagecodecs._imcd" \
--hidden-import="imagecodecs._lzw" \
./utils/postprocess.py
```


The compiled executables can be found in the *dist* directory of the respective project.
Please copy them to the *CTC Submission/SW* directory as shown in 
the desired directory structure above.







