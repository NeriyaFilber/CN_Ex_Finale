# Computer Networking Final Project: Network Traffic Analysis

![Python](https://img.shields.io/badge/Python-3.9.18-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-FF6F00?logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-006400)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-1.5.3-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24.2-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6.3-11557c)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-3776AB)
![Jupyter](https://img.shields.io/badge/Jupyter-1.0.0-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## Table of Contents
- [Executive Summary](#executive-summary)
- [Project Overview](#project-overview)
- [Project Goals](#project-goals)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
- [Project Structure](#project-structure)
  - [PDF Files](#pdf-files)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Data Files](#data-files)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [Conclusion](#conclusion)
- [References](#references)
- [Authors](#authors)

## Executive Summary

This project explores network traffic analysis techniques to identify applications from their traffic patterns, even when encrypted. Using machine learning models including XGBoost and neural networks, we successfully classified different types of network traffic (audio streaming, video streaming, video conferencing, and web browsing) with high accuracy. Our findings demonstrate that network traffic contains distinct patterns that can be leveraged for application identification, which has significant implications for network optimization, security monitoring, and privacy research. The project serves as a comprehensive exploration of modern traffic analysis techniques in an increasingly encrypted internet landscape.

## Project Overview

This repository contains the final project for the Computer Networking course in Ariel University. The project focuses on understanding the characteristics of application network traffic. By analyzing traffic patterns, we can identify which applications are being used, even when the traffic is encrypted. This has important implications for both network optimization and privacy.

Key applications of traffic analysis include:
- Telecom/internet providers can prioritize traffic to improve user experience
- Security researchers can identify potentially malicious applications
- Privacy researchers can understand how much information leaks even in encrypted traffic
- Network administrators can optimize network resources based on application usage patterns

## Project Goals

- Collect network traffic data from various applications
- Extract meaningful features from raw packet data
- Develop classification methods to identify applications from their traffic patterns
- Analyze how encryption affects the identifiability of applications
- Evaluate the privacy implications of traffic analysis techniques
- Demonstrate practical applications of traffic fingerprinting

## Getting Started

### Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.9.18 (specific version required for compatibility)
- Git (for cloning the repository)
- Wireshark (recommended for viewing .pcap files)
- Sufficient computational resources (TensorFlow requires significant CPU/memory)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NeriyaFilber/CN_Ex_Finale.git
   cd computer-networking-project
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install all required packages using the provided commands:
   ```bash
   python3.9 -m pip install pandas==1.5.3
   python3.9 -m pip install matplotlib==3.6.3
   python3.9 -m pip install seaborn==0.12.2
   python3.9 -m pip install xgboost==2.1.1
   python3.9 -m pip install scikit-learn==1.3
   python3.9 -m pip install tensorflow==2.14.0
   python3.9 -m pip install numpy==1.24.2
   python3.9 -m pip install jupyter==1.0.0
   ```

   > **🚨 Important Note:** TensorFlow may not work properly in virtual machines with limited resources. For optimal performance, we recommend running this project on a physical machine with adequate CPU and memory.

### Dataset Setup

1. Download the HTTPS classification dataset from [Kaggle](https://www.kaggle.com/datasets/inhngcn/https-traffic-classification/data)
2. Place the downloaded `HTTPS-clf-dataset.csv` file in the same directory as the Jupyter notebooks
3. Verify the dataset is correctly placed before running the classification notebook

## Project Structure

### PDF Files

| File                                                                                                                                                                                                                               | Description                                                                    |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `Analyzing_HTTPS_encrypted_traffic_to_ide.pdf`<br/>`Early_Traffic_Classification_With_Encrypted_ClientHello_A_Multi-Country_Study.pdf`<br/>`FlowPic_Encrypted_Internet_Traffic_Classification_is_as_Easy_as_Image_Recognition.pdf` | Research articles analyzed in this project with detailed questions and answers |
| `finale_project_instructions_25.pdf`                                                                                                                                                                                               | Official project requirements and instructions                                 |
| `Answers Part A + B.pdf`                                                                                                                                                                                                           | Comprehensive answers to theoretical questions and article analyses            |

### Jupyter Notebooks

| File                                     | Description                                                                                                                                                                                                                                                   |
|------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Network_trafic_recognition_model.ipynb` | This notebook contains our classification models that analyze the HTTPS dataset. The notebook includes data preprocessing, model implementation (Random Forest, XGBoost, Neural Networks), training procedures, evaluation metrics, and performance analysis. |
| `Network_trafic_analysis.ipynb`          | This notebook contains exploratory data analysis and visualization of various captured network traffic types. It analyzes traffic patterns, feature distributions, and correlation studies across different application categories.                           |

### Data Files

Our data files consist of network packet captures (`.pcap`) and their corresponding extracted flow features (`.csv`) from different application types. All captures were collected using Wireshark under controlled conditions to ensure consistent network environments.

#### Audio Streaming Files
| File                           | Description                                                                                                                                         |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `audio_streaming_spotify.pcap` | Raw network packet captures from Spotify music streaming sessions. Contains complete traffic data including audio transmission and control packets. |
| `audio_streaming_spotify.csv`  | Extracted flow features from Spotify captures including packet sizes, inter-arrival times, throughput, and quality metrics.                         |

#### Video Streaming Files  
| File                           | Description                                                                                                                                 |
|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `video_streaming_youtube.pcap` | Raw network packet captures from YouTube video streaming sessions. Contains complete traffic including video segments and control messages. |
| `video_streaming_youtube.csv`  | Extracted flow features from YouTube captures including bitrates, buffer levels, segment sizes, and quality switches.                       |

#### Video Conferencing Files
| File                                      | Description                                                                                                                      |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `video_conferencing_zoom.pcap`            | Raw network packet captures from Zoom video conferencing sessions. Contains traffic for video, audio, and screen sharing.        |
| `video_conferencing_zoom.csv`             | Extracted flow features from Zoom captures including video/audio metrics, participant data, and connection quality.              |
| `video_conferencing_google_meet.pcap`     | Raw network packet captures from Google Meet video conferencing sessions. Contains traffic for video, audio, and screen sharing. |
| `video_conferencing_zoom_google_meet.csv` | Extracted flow features from Google Meet captures including video/audio metrics, participant data, and connection quality.       |

#### Web Surfing Files
| File                      | Description                                                                                                                         |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `web_surfing_chrome.pcap` | Raw network packet captures from Chrome web browsing sessions. Contains HTTP(S) requests, responses, and web content transfers.     |
| `web_surfing_chrome.csv`  | Extracted network flow features from Chrome captures including packet sizes, inter-arrival times, HTTP methods, and response codes. |
| `web_surfing_edge.pcap`   | Raw network packet captures from Edge web browsing sessions. Contains HTTP(S) requests, responses, and web content transfers.       |
| `web_surfing_edge.csv`    | Extracted network flow features from Edge captures including packet sizes, inter-arrival times, HTTP methods, and response codes.   |

#### Classification Dataset
| File                    | Description                                                                                                                                                                                                                                                                                                                                                  |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HTTPS-clf-dataset.csv` | External dataset containing network traffic data from various applications with labeled application types. **Must be downloaded separately** from [Kaggle](https://www.kaggle.com/datasets/inhngcn/https-traffic-classification/data). Notice the data contain only Bytes data that attacker can see, and the type of the network traffic to build the model |

## Key Findings

Our analysis revealed several important discoveries about network traffic patterns:

1. **Application Fingerprinting Effectiveness**: We achieved 99.62% accuracy in classifying applications based solely on their encrypted traffic patterns using our optimized XGBoost model.

2. **Feature Importance**: The most discriminative features for classification were:
   - Packet size distribution
   - Inter-arrival time patterns
   - Flow duration and throughput metrics
   - Protocol-specific characteristics (e.g., UDP in audio streaming, TCP and QUIQ in web browsing)
   - Network layer information (IP addresses)

3. **Encryption Impact**: Despite encryption protecting content, substantial metadata and behavioral patterns remain visible and identifiable.

4. **Model Comparison**: Our experiments showed that while neural networks performed well (96% accuracy), traditional machine learning approaches like XGBoost (99.62%) and Random Forest (99.1%) provided excellent results with faster training times.

5. **Privacy Implications**: Our results demonstrate that even with HTTPS/TLS encryption, user activities remain potentially identifiable through traffic analysis techniques.

## Methodology

Our approach followed these key steps:

1. **Data Collection**: We captured real network traffic from various applications using Wireshark in controlled environments to ensure consistency.

2. **Feature Extraction**: We processed raw packet data to extract meaningful features including:
   - Statistical measures (mean, median, variance) of packet sizes and inter-arrival times
   - Flow-level metrics (duration, bytes transferred, packets per second)
   - Protocol-specific features (TCP window sizes, TLS handshake patterns)

3. **Model Development**: We implemented and compared several classification approaches:
   - Random Forest with hyperparameter optimization
   - XGBoost with feature selection and cross-validation
   - CNN-based classification inspired by the FlowPic approach
   - KNN and SVM for baseline comparisons
   - Logistic Regression for interpretability
   - Ensemble methods for robustness

4. **Evaluation**: We assessed model performance using stratified k-fold cross-validation and independent test sets to ensure robust evaluation.

5. **Analysis**: We examined feature importance, model behavior, and the privacy implications of our findings.

## Conclusion

This project demonstrates that network traffic analysis can effectively identify applications even when traffic is encrypted. Our models achieved high accuracy in classifying different application types, with our optimized XGBoost model achieving over 99.62% accuracy across multiple application categories.

The strong classification performance highlights both opportunities and concerns:

1. **Network Management**: Our findings enable more intelligent traffic prioritization and quality-of-service implementations without requiring deep packet inspection.

2. **Security Applications**: These techniques can help identify anomalous or potentially malicious traffic patterns even when encrypted.

3. **Privacy Considerations**: Our results raise important questions about user privacy in encrypted communications, as significant behavioral information remains observable through traffic analysis.

Future work could explore more advanced deep learning architectures, investigate adversarial techniques to enhance privacy by obfuscating traffic patterns, and expand the range of applications analyzed. Additionally, examining how these techniques perform across different network conditions and encryption protocols would provide valuable insights.

## Submission Notes
- In our project there is not `/src/` and `/res/` directory's, as we used Jupyter notebooks for all code.
- We upload the .pcap file because we have GitHub student account, and we can upload larger files.
- We didn't rum model with IP addresses and port numbers because in our data there isn't any IP addresses and port numbers.
- Our code work on Linux, but it may not work on virtual machines with limited resources because `tensorflow` module requires significant CPU/memory.
- We use GitHub copilot plugin in Pycharm for documentation and improve the explanation english level, this plugin cant export prompts.
- Our model not fitted to regular wireshark captures, because we use our own data, and we have different features. if you want to use our model you need to use our data, or data that have the same 88 features (You can find the features list in [kaggle](https://www.kaggle.com/datasets/inhngcn/https-traffic-classification/data).
## References

- [Analyzing HTTPS encrypted traffic to identify applications](https://ieeexplore.ieee.org/document/8013420)
- [Early Traffic Classification With Encrypted ClientHello: A Multi-Country Study](https://ieeexplore.ieee.org/document/10697158)
- [FlowPic: Encrypted Internet Traffic Classification is as Easy as Image Recognition](https://ieeexplore.ieee.org/document/8845315)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
## Authors

| Name          | GitHub                                           | LinkedIn                                                              |
|---------------|--------------------------------------------------|-----------------------------------------------------------------------|
| Neriya Filber | [Neriya Filber](https://github.com/NeriyaFilber) | [Neriya Filber](https://www.linkedin.com/in/neriya-filber-b67a872a5)  |
| Itay Segev    | [Itay Segev](https://github.com/itaysegev1)      | [Itay Segev](https://www.linkedin.com/in/itaysegev1)                  |
| Salome Timsit | [Salome Timsit](https://github.com/salometimsit) | [Salome Timsit](https://www.linkedin.com/in/salome-timsit-15533330a/) |
| Taliya Cohen  | [Taliya Cohen](https://github.com/Ttalyacohen)   | [Taliya Cohen](https://www.linkedin.com/in/talya-cohen-659054354/)    |

The teacher of the course is [Professor Amit Zeev Dvir](https://cris.ariel.ac.il/en/persons/amit-zeev-dvir).

---

*Created as part of the Computer Networking course at Ariel University, 2025.*

*© 2025 Neriya Filber, Itay Segev, Salome Timsit, Taliya Cohen. All rights reserved.*
