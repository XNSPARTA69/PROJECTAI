# Project AI

This repository contains a project that uses MONAI and YOLOv8 models for image processing.

## Project Structure

- **app.py**: Main file to run the application.
- **create_monai_model.py**: Script to create and train a MONAI model.
- **main.py**: Auxiliary file for project logic.
- **monai_model.pth**: Pre-trained MONAI model.
- **yolov8n.pt**: Pre-trained YOLOv8 model.
- **static/**: Contains static files like JavaScript and CSS.
  - `script.js`: Client-side logic.
  - `styles.css`: Interface styles.
- **templates/**: Contains HTML templates.
  - `index.html`: Main page of the application.
- **uploads/**: Folder for uploading and processing images.
  - Example files included.

## Requirements

- Python 3.8 or higher
- Required libraries (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/XNSPARTA69/PROJECTAI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd PROJECTAI
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python app.py
```

## Contributions

Contributions are welcome. Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
