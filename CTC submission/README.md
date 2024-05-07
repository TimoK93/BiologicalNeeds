# Create CTC Submission Scripts

Our method was submitted to the **7th ISBI Cell Tracking Challenge** as team **ctc741**.
To simplify the submission process, compiled the repository in some binary files
that can be run without the need to install the dependencies.

If you want to run the compiled files, you can download them from the following links:
[To be Done](...)

Furthermore, you can create the submission directory yourself by following the instructions below.

## Compile the Executables

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
cd $ROOT/mht
export PYTHONPATH=$PYTHONPATH:$PWD
cd $ROOT
pyinstaller --onefile --clean ./mht/inference.py
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
./mht/postprocess.py
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
./embedtrack/infer/infer_ctc_data.py
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
./mht/interpolate.py
```

The compiled executables can be found in the *dist* directory of the respective project.
Please copy them to the *CTC Submission/SW* directory as shown in 
[Resulting Submission directory structure](#resulting-submission-directory-structure).


## Copy models and data

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
cp -r $ROOT/Data/train $ROOT/CTC\ Submission/
```

or

```bash
cp -r $ROOT/Data/challenge $ROOT/CTC\ Submission/
```

## Resulting Submission directory structure


The resulting submission directory should have the following structure:

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
│       ├───inference
│       ├───postprocess
│       ├───interpolate
│   ├───BF-C2DL-HSC_01.sh
│   ├───BF-C2DL-HSC_02.sh
│   ├───...
├───BF-C2DL-HSC
│   ├───01
│   │   ├───001.tiff
│   │   ├───...
│   ├───02
│   │   ├───001.tiff
│   │   ├───...
│───...

```

