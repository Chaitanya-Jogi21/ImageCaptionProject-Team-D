Image Captioning Project using BLIP  
Team-D | Major Project

An AI-powered system that automatically generates meaningful captions for images using the **BLIP (Bootstrapping Language-Image Pre-training)** model from HuggingFace.  
This project integrates **Flask**, **Python**, **Transformers**, and a clean web interface to help users upload an image and instantly generate a high-quality caption.

---

## 📌 Table of Contents
1. Project Overview  
2. Why BLIP Instead of CNN + LSTM?  
3. Objectives  
4. Features  
5. Architecture  
6. Tech Stack  
7. Project Structure  
8. Installation & Setup  
9. How to Run  
10. Output Examples  
11. Screenshots  
12. Team Members  
13. Future Enhancements  
14. License  

---

## 📌 1. Project Overview
Image captioning is the task of generating a textual description from an image.  
Traditionally, this is achieved using:

- **CNN** → for feature extraction  
- **LSTM** → for sequence generation  

However, modern Vision-Language Models like **BLIP** outperform the old approach and provide near human-level caption generation.

In this project, BLIP acts as both the **encoder** and **decoder**, eliminating the need to manually build CNN + LSTM pipelines.

---

## 📌 2. Why BLIP Instead of CNN + LSTM?
Although the project report mentions CNN + LSTM, the team selected BLIP for the following reasons:

### ✔ BLIP Advantages  
- **Already trained on millions of image–text pairs**  
- **More accurate than CNN + LSTM**  
- **Faster caption generation**  
- **Does not require huge datasets for training**  
- **Uses transformer-based architecture**  

 ✔ Conceptually the Same Workflow  
- BLIP's image encoder ≈ CNN  
- BLIP's text decoder ≈ LSTM  

So even though we didn’t manually build CNN + LSTM,  
**the underlying encoder-decoder process remains the same**, just upgraded with modern AI.

---

## 🎯 3. Objectives
- Generate meaningful captions for images.  
- Use a modern deep learning model for captioning.  
- Create a simple and elegant web interface.  
- Deploy the system with Flask.  
- Learn practical AI model integration.  

---

## ⭐ 4. Features
- Upload any image  
- Generate accurate captions  
- Clean, responsive UI  
- Fast processing  
- Uses state-of-the-art BLIP model  
- Easy to extend or deploy  

---

## 🧠 5. Architecture (High-Level)
        User Uploads Image
             ↓
 Flask Backend Receives Image
             ↓
 BLIP Processor Extracts Features
             ↓
BLIP Model Generates Caption (Decoder)
             ↓
   Caption Returned to Frontend
             ↓
     User Sees Final Output

---

## 🛠️ 6. Tech Stack

| Category | Technology |
|---------|-------------|
| **AI Model** | BLIP (HuggingFace Transformers) |
| **Backend** | Python, Flask |
| **Frontend** | HTML5, CSS3 |
| **Libraries** | torch, transformers, pillow |
| **Version Control** | Git & GitHub |

---

## 📂 7. Project Structure

image-caption-project/
│── app.py
│── requirements.txt
│── README.md
│
├── templates/
│ └── index.html
│
└── static/
└── style.css


---

🔧 8. Installation & Setup
Step 1 — Clone the Repo 
Step 2 — Install Dependencies  
Step 3 — Run Flask App  
Step 4 — Open Browser  

---

## ▶️ 9. How to Use  
1. Run the project  
2. Go to the website  
3. Upload any image  
4. Click **Generate Caption**  
5. Model returns a meaningful caption  

---

## 📸 10. Output Examples  
Input Image → A dog in a field
Model Output → "a dog running through a grassy area"

Input Image → A bowl of fruits
Model Output → "a bowl filled with apples and bananas"

## 👥 12. Team Members:-
Chaitanya Jogi
Deepak Choudhary
Manoj kumar Yanamadala
Mohd Mudabbir Arafat
Siva Adapa
Anantha Sathish Kumar Palchuri
Sowmiya S
Syed Hasanuddin
Kukatla Kamal

---

## 🚀 13. Future Enhancements
- Add voice narration  
- Deploy on cloud (AWS/Render)  
- Add dataset-based training module  
- Add attention visualization  
- Multi-language caption output  
- Add image-to-story generator  

---

## 📜 14. License  
This project is developed for academic purposes by **Team-D**.  
Feel free to fork and improve it.

---

