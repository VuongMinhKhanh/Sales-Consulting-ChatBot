# Sales-Consulting-ChatBot

## My 12-minute presentation for ChatBot integration for business
- In this video, I cover current problems with SMEs, why using ChatBot (with simple numeric survey), ChatBot's EDA, and SWOT.
- Video link: https://www.youtube.com/watch?si=LckEf59Mguua2EEn&v=OJ2hUt6nVuk&feature=youtu.be

## Introduction
The **Sales Consulting ChatBot** project aims to develop an automated sales consulting tool to assist businesses in managing customer interactions and optimizing sales processes. This ChatBot leverages advanced artificial intelligence (AI) technologies, including large language models (LLMs) and reinforcement learning algorithms, to provide accurate information and deliver an enhanced user experience.

## Key Features
- **Automated Sales Consulting:** The ChatBot operates 24/7, handling multiple customer inquiries simultaneously without delays.
- **Natural Language Processing (NLP):** Utilizes large language models (LLMs) such as ChatGPT to understand and respond to a wide range of user queries naturally and accurately.
- **Reinforcement Learning:** Integrates a basic RL method with Excel data to enable the ChatBot to learn and improve its performance based on user feedback.
- **Real-Time Data Updates:** Ensures that the ChatBot's responses are always accurate and relevant, reflecting any changes in data and business strategies.

## Support Code
We needed some kinds of EDA, customized data crawling, data preprocessing and optimization before actually building our bot, please refer the code in Resource -> Support Code or our visualizations in Google Colab:
- EDA: https://colab.research.google.com/drive/1XXaIzqBGHrAjb1ER4miZO2r-oNSrYfsG?usp=sharing
- Customized data crawling with BeautifulSoup4: https://colab.research.google.com/drive/17lIrxRWQwxX1P5r1f0IAiRZZVJRkBgtG?usp=sharing
- Data preprocessing with Pandas: https://colab.research.google.com/drive/148nOdHIfkQeXkwffxLmNwsPfaoBtz-ei?usp=sharing
- Chunk size optimization: https://colab.research.google.com/drive/1GKoPyU2Vuf6XKSD8efk0UIv7lxB14ixy?usp=sharing

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/VuongMinhKhanh/Sales-Consulting-ChatBot.git sales-consulting-chatbot
    cd sales-consulting-chatbot
    ```
2. **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the ChatBot:**
    ```bash
    python chatbot.py
    ```

## Usage
Once running, you can interact with the ChatBot through command prompt (venv activated): python chatbot.py. The ChatBot's localhost will be shown and you can interact inquiries and receive sales consulting information quickly and accurately.

## Future Improvements
Future enhancements will focus on expanding the product database, improving the language model's accuracy, and integrating additional features like user behavior analysis to further enhance the customer experience

## Contributions
We welcome contributions from the community. Please create a pull request or open an issue if you would like to contribute to this project.

---

© 2024 Sales Consulting ChatBot Team. All rights reserved.
