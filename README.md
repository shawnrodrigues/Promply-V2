cuda was installed and  also https://visualstudio.microsoft.com/visual-cpp-build-tools/

pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install --force-reinstall .\llama_cpp_python-0.2.26+cu122-cp311-cp311-win_amd64.whl



py -3.11 -m venv myenv
myenv\Scripts\activate
pip install -r requirements.txt

deactivate

Remove-Item -Path "P:\Promply-V2\myenv" -Recurse -Force


pip install -r requirements.txt


make a folder called as the uploads










🩷 1️⃣ Open the file

P:\Promply-V2\myenv\Scripts\Activate.ps1
in your editor (VS Code, Notepad++, …).

🩷 2️⃣ Add Tesseract to PATH
At the end of the file, add this line:


$env:PATH = "C:\Program Files\Tesseract-OCR;$env:PATH"

deactivate
.\myenv\Scripts\activate


tesseract --version





-----read me----
the .env file

# Google Custom Search
GOOGLE_API_KEY=
GOOGLE_CX=

# Gemini
GEMINI_API_KEY=

# OpenAI 
OPENAI_API_KEY=
