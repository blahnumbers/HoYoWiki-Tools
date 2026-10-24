# HoYoWiki-Tools

A collection of small applications to streamline the work of HoYoWiki collaborators.

![hoyowiki_tools_thumbnail](https://github.com/user-attachments/assets/d73a15d9-5240-47c4-8074-755a99406683)

<details>

<summary>UI Preview & Functionalities</summary>

![hoyowiki_tools_streamlit](https://github.com/user-attachments/assets/e5500382-8272-4690-a017-5f2e4aef3678)

### Currently available modules

* **PGC Creator:** Creates and fills in a PGC spreadsheet in batch from a template for it to be loaded and pushed into the WET.
* **Quest Formatter:** Takes text data from a localization sheet made by the Quest Team and formats it so it is ready to be pasted directly into the WET. Supports pasting the text columns or importing the original `.xlsx` / `.csv` data file, including automatic localization of speaker names, locations and field labels from the file's `Translation` section.

</details>

## Local Installation (Windows, macOS, Linux)

Follow the steps below if you want to run HoYoWiki-Tools locally on your own machine.

### Step 1: Install Python

To run HoYoWiki-Tools locally, you will need [Python](https://www.python.org/downloads/) installed on your computer. You can check for an existing installation by running the following command in a terminal:
```sh
python --version
```

> [!NOTE]
> HoYoWiki-Tools has been developed and tested on `v3.13.7` of Python.

### Step 2: Download the latest version of the app's code

Here, you can choose between two methods:

1. If you have [Git](https://www.git-scm.com/) installed on your computer, you can clone this repository by running `git clone https://github.com/Antasma245/HoYoWiki-Tools.git` in the folder where you want the code to be stored.

2. On the main page of the repository, go to `Code` and press `Download ZIP` (or click [here](https://github.com/Antasma245/HoYoWiki-Tools/archive/refs/heads/main.zip)). Then, extract the downloaded ZIP archive where you want the code to be stored.

### Step 3: Set up a virtual environment

*This step is optional but highly recommended.*

In the folder where you extracted the app's code, open a terminal and run the following commands:

#### Step 3a: Initialize virtual environment
```sh
python -m venv venv
```

#### Step 3b: Activate virtual environment
For Windows users:
```sh
venv\Scripts\activate
```

> [!TIP]
> If you get an error saying the execution of scripts is disabled on your system, run the following command and try again.
> ```sh
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

For macOS/Linux users:
```sh
source venv/bin/activate
```

### Step 4: Install requirements

In the same terminal, run:
```sh
python -m pip install -r requirements.txt
```

### Step 5: Run the app

Finally, launch the app by running:
```sh
python -m streamlit run app.py
```

<details>

<summary>Updating your Installation</summary>

1. If you installed HoYoWiki-Tools using Git, open a terminal in the folder where you extracted the app's code and run `git pull`. Then, follow the installation steps starting from **Step 3b**.

2. If you installed HoYoWiki-Tools manually, follow the installation steps from the beginning to get a new version of the app you will put in a new folder (don't forget to delete the other folder containing the old installation afterwards).

</details>

## Developer notes

This application is meant to be used by HoYoWiki collaborators, but is by no means officially affiliated or endorsed by HoYoverse.

### Running the tests

With the virtual environment from the installation steps activated, run:

```sh
python -m unittest discover -s tests
```

This program uses the Streamlit library, which is open sourced under the Apache 2.0 license. A copy of the aforementioned license document can be found in the [`appendix`](appendix) folder of the application or on Streamlit's [GitHub page](https://github.com/streamlit/streamlit?tab=Apache-2.0-1-ov-file).
