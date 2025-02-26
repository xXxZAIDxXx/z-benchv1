
# Z_Bench

## 📌 

 هو برنامج بسيط لكن قوي لتحليل ومقارنة ردود الذكاء الاصطناعي على سؤال محدد، استخدمت نفس السؤال  على عدة نماذج ذكاء اصطناعي، ثم جمعت الردود في هذا البرنامج. الفكرة أنك تطرح سؤالًا (ويُفضّل أن يكون باللغة الإنجليزية لزيادة الدقة) على عدة نماذج ذكاء اصطناعي، ثم يقوم البرنامج بتحليل الردود، واحتساب مجموعة من المعايير، ثم يعرض النتائج في مخططات بيانية.
"**What is Happiness?**"

### 🔹 الميزات والوظائف

1. **📊 الدقة (Correctness):**  
   - يعتمد على مكتبة **spaCy** (نموذج `en_core_web_lg`) لحساب مدى التشابه الدلالي بين السؤال والرد.  
   - يصنف الرد بأنه "دقيق" إذا كان التشابه أعلى من **0.65**، و"غير دقيق" إذا كان أقل من **0.35**، و"جزئي" إذا كان بين القيمتين.

2. **🔢 طول الرد (Length):**  
   - يحسب عدد الكلمات في الرد باستخدام توكنيزر spaCy.

3. **📝 التعقيد (Complexity):**  
   - يحسب متوسط عدد الحروف لكل كلمة، وهذا يعطي مؤشرًا على مدى بساطة أو تعقيد الرد.

4. **📌 تحليل المشاعر (Sentiment Analysis):**  
   - يستخدم نموذج **DistilBERT** (من مكتبات **Transformers** و **Torch**) المصمم على SST-2 لتصنيف الرد إلى **إيجابي**، **سلبي**، أو **حيادي**.  
   - يعتمد على العتبات (>0.65) لتحديد التصنيف.

5. **🔗 صلة الرد (Relevance):**  
   - يحول التشابه بين السؤال والرد إلى نسبة مئوية (0–100%). كلما زادت النسبة، زادت الصلة بالسؤال الأصلي.

6. **📈 توزيع الدقة (Correctness Distribution):**  
   - يحسب نسبة الردود التي صنفت على أنها **دقيق، جزئي، أو غير دقيق**، ويعرضها في مخطط خاص.

---

### 📦 المكتبات المطلوبة

- **Python 3.x**
- **Matplotlib**
- **spaCy** (وتشغيل `python -m spacy download en_core_web_lg`)
- **Transformers** + **Torch** (لتشغيل DistilBERT)
- **NumPy**
- **TextBlob** (اختياري)
- **arabic_reshaper** و **python-bidi** (لعرض النصوص العربية بشكل صحيح)

---

### ⚙️ طريقة الاستخدام

#### ✅ **تنصيب المكتبات** (مثال):
```bash
pip install matplotlib spacy transformers torch numpy textblob arabic-reshaper python-bidi
python -m spacy download en_core_web_lg
```

#### **🔹 Recommended: Using a Virtual Environment**
It's best to use a virtual environment to keep dependencies organized.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

💡 **To deactivate the virtual environment**, use:
```bash
deactivate
```

#### 🚀 **تشغيل البرنامج:**
```bash
python z_benchv1.py
```

#### 💻 **واجهة المستخدم:**
- أدخل **اسم النموذج (AI Model Name)**.  
- أدخل **السؤال (يفضل باللغة الإنجليزية)**.  
- الصق **الردود الناتجة من عدة ذكاءات اصطناعية (واحد في كل سطر)**.  
- اضغط **"Analyze & Submit"** لعرض النتائج في القائمة.  
- اضغط **"Compare Results"** لعرض **المخططات البيانية**:
  - **📊 المقاييس الرئيسية**  
  - **📊 توزيع المشاعر**  
  - **📊 توزيع الدقة**  
- اضغط **"Save Graph"** لحفظ المخطط بصيغة **PNG**.  
- اضغط **"Quit"** للخروج.

---

## 🌍 

**Z_Bench** is a straightforward yet robust Python tool designed to analyze and compare AI-generated responses to a specific question. the question **"What is Happiness?"** was posed to multiple AI models, and their responses were gathered within this program. The concept involves submitting a single question—preferably in English for higher accuracy—to several AI models, evaluating each response based on multiple metrics, and then visualizing the results in clear bar charts.

### 🔹 Features & Functionality

1. **📊 Correctness**  
   - Uses **spaCy** (`en_core_web_lg`) to compute semantic similarity between the question and each response.  
   - Labels a response as **"correct"** if similarity > **0.65**, **"incorrect"** if < **0.35**, and **"partial"** otherwise.

2. **🔢 Response Length**  
   - Counts the number of words in each response using spaCy’s tokenizer.

3. **📝 Complexity**  
   - Calculates the average number of characters per word, indicating the text’s simplicity or complexity.

4. **📌 Sentiment Analysis**  
   - Employs **DistilBERT** (via **Transformers** and **Torch**), fine-tuned on SST-2, to classify responses as **positive, negative, or neutral**, using probability thresholds (>0.65).

5. **🔗 Relevance**  
   - Converts the semantic similarity score between the question and response into a percentage (0–100%).

6. **📈 Correctness Distribution**  
   - Aggregates the percentage of responses labeled as **correct, partial, or incorrect**, and displays them in a dedicated chart.

---

### 📦 Required Libraries

- **Python 3.x**
- **Matplotlib**
- **spaCy** (and run `python -m spacy download en_core_web_lg`)
- **Transformers** + **Torch** (for **DistilBERT** sentiment analysis)
- **NumPy**
- **TextBlob** (optional)
- **arabic_reshaper** and **python-bidi** (for proper Arabic text rendering)

---

### ⚙️ Usage

#### ✅ **Install Dependencies** :
```bash
pip install matplotlib spacy transformers torch numpy textblob arabic-reshaper python-bidi
python -m spacy download en_core_web_lg
```

#### 🚀 **Run the Program:**
```bash
python z_benchv1.py
```

#### 💻 **GUI Workflow:**
- Enter an **AI model name**.  
- Enter a **question (English recommended)**.  
- Paste multiple **AI responses (one per line)**.  
- Click **"Analyze & Submit"** to see metrics in the listbox.  
- Click **"Compare Results"** to view **three bar charts**:
  - **📊 Main Metrics**
  - **📊 Sentiment Distribution**
  - **📊 Correctness Distribution**  
- Click **"Save Graph"** to export the figure as a **PNG**.  
- Click **"Quit"** to exit.

