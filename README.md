# SafeSchool: Real-time Weapon Detection using YOLOv8 with LangChain-based Threat Analysis Agent

## Overview
SafeSchool is an innovative security surveillance system that leverages artificial intelligence to enhance threat detection and response capabilities in educational environments. The system combines YOLOv8's real-time object detection with a LangChain-powered AI agent for continuous, automated threat monitoring and analysis. By replacing traditional human monitoring with AI-driven analysis, SafeSchool significantly reduces response times while maintaining consistent surveillance effectiveness.

## Key Features
- Real-time weapon detection using YOLOv8 with 99% accuracy
- Automated threat analysis through LangChain-based AI agent
- Location-aware security alerts based on camera positioning
- Interactive analysis interface with carousel display
- Multi-frame threat confirmation for reduced false positives
- Response time under 6.2 seconds from detection to alert

## Dataset
This implementation utilizes the mock attack dataset from the research paper "Real-time gun detection in CCTV: An open problem" (Salazar González et al., 2020). The dataset comprises 5,149 annotated frames collected from three strategic camera positions in a university environment, capturing various lighting conditions and scenarios.

Dataset specifications:
- Total annotated frames: 5,149
- Frame rate: 2 FPS
- Camera perspectives: Three distinct locations
- Weapon classes: Handgun, Knife, Short_rifle

### Download and Setup
1. Download the dataset from the following link:  
   [Weapons Images Dataset (2fps)](https://uses0-my.sharepoint.com/personal/jsalazar_us_es/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fjsalazar%5Fus%5Fes%2FDocuments%2FShared%2FVICTORY%2FUS%2Fweapons%5Fimages%5F2fps%2Ezip&parent=%2Fpersonal%2Fjsalazar%5Fus%5Fes%2FDocuments%2FShared%2FVICTORY%2FUS&ga=1)  

2. Extract the contents of the ZIP file.  

3. Copy the `Images` folder and paste it inside the `data/` directory:
   ```bash
   mv path_to_extracted_folder/Images safeschool/data/

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU recommended
- OpenAI API key for LangChain integration

### Environment Setup
```bash
# Clone repository
git clone https://github.com/narasimhakarthik2/SafeSchool.git
cd safeschool

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration
1. Set up environment variables:
```bash
export OPENAI_API_KEY=your_api_key
```

2. Update config.yaml with desired settings:
```yaml
model:
  name: 'yolov8n'
  classes: ['Handgun', 'Knife', 'Short_rifle']
  confidence_threshold: 0.75
  iou_threshold: 0.45

data:
  processed_dir: '/data/mock_attack_dataset'
  img_size: 640
  augmentation: true

training:
  batch_size: 8
  epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0005
```

## Project Structure
```
safeschool/
├── data/
│   └── mock_attack_dataset/
├── model/
│   └── best.pt
│── training/
│   └── trainer.yaml
├── utils/
│   ├── weapon_detector.py
│   └── camera_mapping.py
├── analysis/
│   └── llm_agent.py
├── configs/
│   └── model_config.yaml
├── main.py
└── train.py
```

## Usage Instructions
### Clean and organize the dataset
```bash
python data_organizer.py
```
### Training
To train the YOLOv8 model on your dataset:
```bash
python train.py
```

### Running Detection
For real-time weapon detection and threat analysis:
```bash
python main.py
```

### Interface Controls
- 'Q': Exit application
- 'A': Navigate to previous analysis
- 'D': Navigate to next analysis

## Performance Metrics
- Detection Speed: <50ms per frame
- Threat Analysis Time: ~6 seconds
- Detection Accuracy: >90% for firearms
- False Positive Rate: <1%

## Citation
If you use this implementation or the mock attack dataset, please cite:
```bibtex
@article{SalazarGonzalez2020,
title = "Real-time gun detection in CCTV: An open problem",
journal = "Neural Networks",
year = "2020",
doi = "https://doi.org/10.1016/j.neunet.2020.09.013",
author = "Salazar Gonz{\'{a}}lez, Jose L. and Zaccaro, Carlos and {\'{A}}lvarez-Garc{\'{i}}a, Juan A. and Soria-Morillo, Luis M. and Sancho Caparrini, Fernando"
}
```

## Requirements
- Python 3.8+
- PyTorch >= 1.7
- OpenCV
- Ultralytics
- LangChain
- OpenAI API access

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or collaboration:
[Your contact information]

## Disclaimer
This system is designed for research and security enhancement purposes only. Users are responsible for compliance with local regulations regarding surveillance and security systems.