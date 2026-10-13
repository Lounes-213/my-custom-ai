# 🤖 Custom AI Assistant (Fine-Tuning with QLoRA)

مشروع متكامل لبناء وتطوير نموذج ذكاء اصطناعي مخصص خاص بك (Custom LLM)، وتدريبه في **Google Colab (T4 GPU)** وحفظه وتشغيله.

---

## 📁 بنية المشروع (Project Structure)

```text
├── data/
│   └── dataset.json         # قاعدة بيانات الأسئلة والأجوبة الخاصة بالـ AI ديالك
├── output/                  # المجلد الذي يُحفظ فيه النموذج بعد التدريب (تلقائياً)
├── requirements.txt         # المكتبات المطلوبة (Transformers, PEFT, TRL, etc.)
├── train.py                 # سكربت التدريب السريع بتقنية QLoRA 4-bit
├── chat.py                  # سكربت الدردشة والتحدث مع النموذج
├── colab_starter.ipynb      # نوت بوك جاهز للتشغيل مباشرة في Google Colab
├── .gitignore               # استثناء الملفات الكبيرة من الرفع إلى GitHub
└── README.md                # دليل الاستخدام
```

---

## 🛠️ 1. كيفاش تزيد البيانات الخاصة بيك (Custom Data)

ادخل لملف `data/dataset.json` وزيد أي أسئلة وأجوبة تحب الـ AI يتعلمها بالشكل التالي:

```json
[
  {
    "instruction": "من أنت؟",
    "input": "",
    "output": "أنا المساعد الذكي الخاص بك!"
  },
  {
    "instruction": "Chkoun nta?",
    "input": "",
    "output": "Ana l-AI l-khas bik!"
  }
]
```

---

## 🚀 2. كيفاش ترفع التحديثات لـ GitHub (Push to GitHub)

كل ما تزيد بيانات أو تعدل في الكود، نفّذ هذا السطر فقط:

```bash
git add .
git commit -m "update data"
git push
```

---

## ⚡ 3. كيفاش تشغلو وتدربو في Google Colab

1. افتح [Google Colab](https://colab.research.google.com).
2. تأكد أن الـ GPU مفعل: **`Exécution` ⬅️ `Modifier le type d'exécution` ⬅️ `T4 GPU`**.
3. انسخ هذا الكود في خلية جديدة في Colab واضغط **Play**:

```bash
# 1. سحب المشروع
!git clone https://github.com/Lounes-213/my-custom-ai.git
%cd my-custom-ai

# 2. تثبيت المكتبات
!pip install -q -r requirements.txt

# 3. إطلاق التدريب على كارت الشاشة GPU
!python train.py --epochs 5 --batch_size 2

# 4. التحدث مع الـ AI ديالك!
!python chat.py --prompt "Chkoun nta?"
```

---

## 💬 4. التحدث مع النموذج التفاعلي (Interactive Chat)

تقدر تهدر معاه في سطر الأوامر مباشرة:

```bash
python chat.py
```
أو تسألو سؤال واحد مباشرة:
```bash
python chat.py --prompt "كيفاش تقدر تعاوني؟"
```
