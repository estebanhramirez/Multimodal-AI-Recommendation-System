# Multimodal-AI-Recommendation-System
Repository with the Python code used for implementing a Multimodal (AI-powered) recommendation system.

## 1. Create the project directory

### 1.1. Create the root folder
Create a folder named `<project_name>`, at some path `~/<prefix_path>` <br>
<br>
**Command `[Command Prompt]`:**
```
mkdir "~/<prefix_path>/<project_name>"
```
**Result `[File directory]`:**
```
<prefix_path>/
        └── <project_name>/
```

### 1.2. Create (local) data folders
Create two folders named `Input_data` and `Output_data`, inside the root folder, `~/<prefix_path>/<project_name>` <br> 
<br>
**Command `[Command Prompt]`:**
```
mkdir "~/<prefix_path>/<project_name>/Input_data"
mkdir "~/<prefix_path>/<project_name>/Output_data"
```
**Result `[File directory]`:**
```
<prefix_path>/
        └── <project_name>/
                ├── Input_data/
                └── Output_data/
```
Make sure the raw ESNcard holders' information, as exported from Google Forms, is stored inside folder `Input_data`. You can change the paths and names in the scripts, in order to make sure that the names are consistent. The cleaned, formatted and pre-processed data will be stored inside folder `Output_data`.


### 1.2. Clone the GitHub repository
Clone this repository to a folder `Repo`, inside the root folder, `~/<prefix_path>/<project_name>` <br>
<br>
**Command `[Command Prompt]`:**
```
git clone "https://github.com/estebanhramirez/ESNcard-info-formatting.git" "~/<prefix_path>/<project_name>/Repo"
```
**Result `[File directory]`:**
```
<prefix_path>/
        └── <project_name>/
                    ├── Input_data/
                    ├── Output_data/
                    └── Repo/
                          ├── README.md
                          └── ...
```

<br>
<br>


## 2. Set-up the virtual environment

### 2.1. Create the virtual environment
Create a new folder name `venv` inside the repository folder, `~/<prefix_path>/<project_name>/Repo`, with an isolated and self-contained virtual environment <br>
<br>
**Command `[Command Prompt]`:**
```
python3.12 -m venv "~/<prefix_path>/<project_name>/Repo/venv"
```

### 2.2. Activate the virtual environment
**Command `[Command Prompt]`:**
```
source "~/<prefix_path>/<project_name>/Repo/venv/bin/activate"
```
or (in Windows)
```
"~\<prefix_path>\<project_name>\Repo\venv\Scripts\activate.bat"
```
We can deactivate the virtual environment using `deactivate.bat`. Please notice that the quotes are important in the path.

<br>
<br>


## 3. Install the required libraries
**Command `[Command Prompt]`:**
```
pip install -r "~/<prefix_path>/<project_name>/Repo/requirements.txt"
```

<br>
<br>


## 4. Set-up OLlama API

### 4.1. Install OLlama
Install OLlama in your local laptop<br>
<br>
**Command `[PowerShell]`:**
```
irm https://ollama.com/install.ps1 | iex
```

### 4.2. Pull Ollama models
Download the model `llama2`<br>
<br>
**Command `[Command Prompt | PowerShell]`:**
```
ollama pull llama2
```
Check that the model was downloaded<br>
<br>
**Command `[Command Prompt | PowerShell]`:**
```
ollama list
```

### 4.3. Start the Ollama HTTP Server/API
**Command `[Command Prompt | PowerShell]`:**
```
ollama serve
```

<br>
<br>


## 5. Set-up LocalTunnel
Once Node.js installed, LocalTunnel lets us create a public link instantly with a single command

### 5.1. Install Node.js®
Install Node.js®'s prebuilt from the official website: [Node.js®](https://nodejs.org/en/download).

### 5.2. Install LocalTunnel
Once Node.js installed, install LocalTunnel
**Command `[Command Prompt]`:**
```
npm install -g localtunnel
```

### 5.3. Run the Python server
Run the FastAPI app to handle prompts and communicate with the local LLM<br>
<br>
**Command `[Command Prompt]`:**
```
python "~/<prefix_path>/<project_name>/Repo/app.py"
