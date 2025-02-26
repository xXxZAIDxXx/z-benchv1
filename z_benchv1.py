import tkinter as tk
from tkinter import messagebox, scrolledtext, Listbox, filedialog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple
import numpy as np
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob  # (Keep if needed elsewhere)

# Arabic reshaping libraries
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    messagebox.showinfo("Dependency Notice", "Installing arabic-reshaper and python-bidi...")
    import subprocess
    subprocess.check_call(["pip", "install", "arabic-reshaper"])
    subprocess.check_call(["pip", "install", "python-bidi"])
    import arabic_reshaper
    from bidi.algorithm import get_display

def ar(txt: str) -> str:
    reshaped = arabic_reshaper.reshape(txt)
    return get_display(reshaped)

class ZBenchv1:
    def __init__(self):
        # Load spaCy English model (for similarity)
        self.nlp = spacy.load("en_core_web_lg")
        # Load a transformer sentiment model (DistilBERT fine-tuned on SST-2)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def evaluate_response(self, question: str, response: str) -> Dict[str, str]:
        """Evaluate the response using spaCy similarity for correctness/relevance and transformers for sentiment."""
        analyses = {}
        # Convert texts to spaCy docs
        doc_q = self.nlp(question)
        doc_r = self.nlp(response)
        
        # 1. Correctness via semantic similarity
        sim_score = doc_q.similarity(doc_r)
        if sim_score > 0.65:
            analyses["Correctness"] = "correct"
        elif sim_score < 0.35:
            analyses["Correctness"] = "incorrect"
        else:
            analyses["Correctness"] = "partial"
        
        # 2. Response Length (word count)
        word_count = len(doc_r)
        analyses["Length (words)"] = str(word_count)
        
        # 3. Complexity (average characters per word)
        if word_count > 0:
            avg_chars = sum(len(token.text) for token in doc_r) / word_count
            analyses["Complexity"] = f"{avg_chars:.2f} (avg chars/word)"
        else:
            analyses["Complexity"] = "0.00"
        
        # 4. Sentiment using transformers
        inputs = self.sentiment_tokenizer(response, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        if probs[1] > 0.65:
            analyses["Sentiment"] = "positive"
        elif probs[0] > 0.65:
            analyses["Sentiment"] = "negative"
        else:
            analyses["Sentiment"] = "neutral"
        
        # 5. Relevance via similarity (scaled to percentage)
        relevance = doc_q.similarity(doc_r) * 100
        analyses["Relevance (%)"] = f"{relevance:.2f}"
        
        return analyses

    def run_benchmark(self, question: str, ai_responses: List[str]) -> Tuple[Dict[str, float], List[Dict[str, str]]]:
        ratings = []
        # Track correctness counts
        correctness_dist = {"correct": 0, "partial": 0, "incorrect": 0}
        for response in ai_responses:
            rating = self.evaluate_response(question, response)
            ratings.append(rating)
            c = rating["Correctness"]
            if c in correctness_dist:
                correctness_dist[c] += 1
        
        metrics = {
            "Correctness Pass Rate (%)": 0.0,
            "Average Length (words)": 0.0,
            "Average Complexity (chars/word)": 0.0,
            "Sentiment Distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "Average Relevance (%)": 0.0,
            "Correctness Distribution": {"correct": 0.0, "partial": 0.0, "incorrect": 0.0}
        }
        
        total = len(ai_responses)
        if total == 0:
            return metrics, ratings
        
        correct_count = sum(1 for r in ratings if r["Correctness"] == "correct")
        metrics["Correctness Pass Rate (%)"] = (correct_count / total) * 100
        
        for label in ["correct", "partial", "incorrect"]:
            metrics["Correctness Distribution"][label] = (correctness_dist[label] / total) * 100
        
        lengths = [float(r["Length (words)"]) for r in ratings]
        metrics["Average Length (words)"] = sum(lengths) / total
        
        complexities = [float(r["Complexity"].split()[0]) for r in ratings]
        metrics["Average Complexity (chars/word)"] = sum(complexities) / total
        
        for r in ratings:
            metrics["Sentiment Distribution"][r["Sentiment"]] += 1
        for k in metrics["Sentiment Distribution"]:
            metrics["Sentiment Distribution"][k] /= total
        
        relevances = [float(r["Relevance (%)"]) for r in ratings]
        metrics["Average Relevance (%)"] = sum(relevances) / total
        
        return metrics, ratings

class ZBenchGUI:
    def __init__(self, root):
        self.root = root
        try:
            self.root.option_add('*Font', ('Noto Sans Arabic', 11))
        except tk.TclError:
            self.root.option_add('*Font', ('Arial', 11))
        self.root.title(ar("ZAID_AI Quick Bench - AI Response Benchmark "))
        self.root.configure(bg='black')
        self.root.rowconfigure(0, weight=0)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=1)
        self.root.rowconfigure(4, weight=1)
        self.root.rowconfigure(5, weight=0)
        self.root.columnconfigure(0, weight=1)
        self.bench = ZBenchv1()
        self.ai_results = {}
        self.question = ""
        self.fig = None
        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(self.root,
            text=ar("ZAID_AI Quick Bench 1.0 - AI Response Benchmark "),
            font=("Noto Sans Arabic", 20, "bold"),
            fg='white', bg='black')
        title_label.grid(row=0, column=0, pady=10, sticky='n')
        
        instructions = tk.Label(self.root,
            text=ar("اكتب سؤالك ثم الصق الردود النصية من الذكاء الاصطناعي وقارن النتائج."),
            font=("Noto Sans Arabic", 14),
            fg='white', bg='black')
        instructions.grid(row=1, column=0, pady=5, sticky='n')
        
        top_frame = tk.Frame(self.root, bg='black')
        top_frame.grid(row=2, column=0, pady=10, sticky='ew')
        top_frame.columnconfigure(0, weight=0)
        top_frame.columnconfigure(1, weight=0)
        top_frame.columnconfigure(2, weight=1)
        top_frame.columnconfigure(3, weight=0)
        top_frame.columnconfigure(4, weight=1)
        
        tk.Label(top_frame, text=ar("AI Model Name / اسم النموذج:"), fg='white', bg='black').grid(row=0, column=0, padx=5, sticky='e')
        self.ai_entry = tk.Entry(top_frame, width=20)
        self.ai_entry.grid(row=0, column=1, padx=5, sticky='w')
        add_ai_button = tk.Button(top_frame, text=ar("Add AI / إضافة نموذج"), command=self.add_ai,
                                  bg='#4CAF50', fg='white', relief='flat', padx=10, pady=5)
        add_ai_button.grid(row=0, column=2, padx=5, sticky='w')
        tk.Label(top_frame, text=ar("Enter Question / أدخل السؤال:"), fg='white', bg='black').grid(row=0, column=3, padx=5, sticky='e')
        self.question_entry = tk.Entry(top_frame, width=25)
        self.question_entry.grid(row=0, column=4, padx=5, sticky='w')
        test_question_button = tk.Button(top_frame, text=ar("Test Question / اختبار السؤال"), command=self.test_question,
                                         bg='#2196F3', fg='white', relief='flat', padx=10, pady=5)
        test_question_button.grid(row=0, column=5, padx=5, sticky='w')
        
        response_frame = tk.Frame(self.root, bg='black')
        response_frame.grid(row=3, column=0, padx=10, pady=10, sticky='nsew')
        response_frame.rowconfigure(0, weight=0)
        response_frame.rowconfigure(1, weight=1)
        response_frame.columnconfigure(0, weight=1)
        tk.Label(response_frame, text=ar("Enter All Raw AI Responses / أدخل جميع الردود:"), fg='white', bg='black').grid(row=0, column=0, sticky='w', pady=5)
        self.response_text = scrolledtext.ScrolledText(response_frame, width=70, height=8, bg='white', fg='black')
        self.response_text.grid(row=1, column=0, sticky='nsew', pady=5)
        analyze_submit_button = tk.Button(response_frame, text=ar("Analyze & Submit / تحليل وإرسال"), command=self.analyze_and_submit,
                                          bg='#F44336', fg='white', relief='flat', padx=20, pady=8)
        analyze_submit_button.grid(row=2, column=0, pady=5, sticky='e')
        
        results_frame = tk.Frame(self.root, bg='black')
        results_frame.grid(row=4, column=0, padx=10, pady=10, sticky='nsew')
        results_frame.rowconfigure(0, weight=0)
        results_frame.rowconfigure(1, weight=1)
        results_frame.columnconfigure(0, weight=1)
        tk.Label(results_frame, text=ar("Analysis Results / نتائج التحليل:"), fg='white', bg='black').grid(row=0, column=0, sticky='w', pady=5)
        self.results_listbox = Listbox(results_frame, width=80, height=10, bg='white', fg='black')
        self.results_listbox.grid(row=1, column=0, sticky='nsew')
        
        button_frame = tk.Frame(self.root, bg='black')
        button_frame.grid(row=5, column=0, pady=10)
        compare_button = tk.Button(button_frame, text=ar("Compare Results / مقارنة النتائج"), command=self.compare_results,
                                   bg='#9C27B0', fg='white', relief='flat', padx=15, pady=8)
        compare_button.pack(side='left', padx=5)
        save_graph_button = tk.Button(button_frame, text=ar("Save Graph / حفظ الرسم"), command=self.save_graph,
                                      bg='#FF9800', fg='white', relief='flat', padx=15, pady=8)
        save_graph_button.pack(side='left', padx=5)
        quit_button = tk.Button(button_frame, text=ar("Quit / الخروج"), command=self.quit_app,
                                bg='#607D8B', fg='white', relief='flat', padx=15, pady=8)
        quit_button.pack(side='left', padx=5)
        
        tk.Label(self.root, text=ar("انسخ ولصق جميع ردود الذكاء الاصطناعي هنا للتحليل."), fg='white', bg='black').grid(row=6, column=0, pady=5, sticky='n')

    def add_ai(self):
        ai_name = self.ai_entry.get().strip()
        if not ai_name:
            messagebox.showerror("Error", "Please enter an AI model name.")
            return
        self.ai_results[ai_name] = None
        self.results_listbox.delete(0, tk.END)
        self.results_listbox.insert(tk.END, f"Added AI: {ai_name}")
        self.ai_entry.delete(0, tk.END)
        self.question_entry.delete(0, tk.END)
        self.response_text.delete(1.0, tk.END)

    def test_question(self):
        self.question = self.question_entry.get().strip()
        if not self.question:
            messagebox.showerror("Error", "Please enter a question to test.")
            return
        self.results_listbox.delete(0, tk.END)
        self.results_listbox.insert(tk.END, f"Testing question: {self.question}")
        self.results_listbox.insert(tk.END, "Paste raw AI responses below and click 'Analyze & Submit'.")

    def analyze_and_submit(self):
        if not self.question or not self.ai_results:
            messagebox.showerror("Error", "Please add an AI and enter a question first.")
            return
        responses_text = self.response_text.get(1.0, tk.END).strip()
        responses = [r.strip() for r in responses_text.split('\n') if r.strip()]
        if not responses:
            messagebox.showerror("Error", "Please enter at least one response.")
            return
        try:
            ai_name = list(self.ai_results.keys())[-1]
            metrics, ratings = self.bench.run_benchmark(self.question, responses)
            self.results_listbox.delete(0, tk.END)
            self.results_listbox.insert(tk.END, f"Results for AI: {ai_name}")
            self.results_listbox.insert(tk.END, f"Question: {self.question}")
            self.results_listbox.insert(tk.END, f"Total Responses: {len(responses)}")
            self.results_listbox.insert(tk.END, f"Correctness Pass Rate (%): {metrics['Correctness Pass Rate (%)']:.2f}")
            self.results_listbox.insert(tk.END, f"Average Length (words): {metrics['Average Length (words)']:.2f}")
            self.results_listbox.insert(tk.END, f"Average Complexity (chars/word): {metrics['Average Complexity (chars/word)']:.2f}")
            self.results_listbox.insert(tk.END, f"Sentiment Distribution: Positive={metrics['Sentiment Distribution']['positive']:.2%}, Neutral={metrics['Sentiment Distribution']['neutral']:.2%}, Negative={metrics['Sentiment Distribution']['negative']:.2%}")
            self.results_listbox.insert(tk.END, f"Average Relevance (%): {metrics['Average Relevance (%)']:.2f}")
            self.results_listbox.insert(tk.END, "\nDetailed Ratings and Analyses:")
            for i, (response, rating) in enumerate(zip(responses, ratings), 1):
                short_resp = response[:50] + ('...' if len(response) > 50 else '')
                self.results_listbox.insert(tk.END, f"  Attempt {i}: '{short_resp}'")
                for metric, value in rating.items():
                    self.results_listbox.insert(tk.END, f"    - {metric}: {value}")
            self.ai_results[ai_name] = metrics
            self.question_entry.delete(0, tk.END)
            self.response_text.delete(1.0, tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def compare_results(self):
        if not self.ai_results or all(v is None for v in self.ai_results.values()):
            messagebox.showerror("Error", "No AI results to compare. Please test at least one AI.")
            return
        models = list(self.ai_results.keys())
        metrics_list = [res for res in self.ai_results.values() if res is not None]
        metric_names = [
            ar("Correctness Pass Rate (%) / نسبة الإجابات الصحيحة"),
            ar("Average Length (words) / متوسط طول الرد"),
            ar("Average Complexity (chars/word) / متوسط التعقيد"),
            ar("Average Relevance (%) / متوسط الصلة")
        ]
        # Build data for left chart (Main metrics)
        model_metric_data = {}
        for model, m in zip(models, metrics_list):
            model_metric_data[model] = [
                m["Correctness Pass Rate (%)"],
                m["Average Length (words)"],
                m["Average Complexity (chars/word)"],
                m["Average Relevance (%)"]
            ]
        # Build data for middle chart (Sentiment Distribution)
        sentiments = [ar("Positive / إيجابي"), ar("Neutral / محايد"), ar("Negative / سلبي")]
        model_sentiment_data = {}
        for model, m in zip(models, metrics_list):
            pos = m["Sentiment Distribution"]["positive"] * 100
            neu = m["Sentiment Distribution"]["neutral"] * 100
            neg = m["Sentiment Distribution"]["negative"] * 100
            model_sentiment_data[model] = [pos, neu, neg]
        # Build data for right chart (Correctness Distribution)
        correctness_labels = ["correct", "partial", "incorrect"]
        model_correctness_data = {}
        for model, m in zip(models, metrics_list):
            c = m["Correctness Distribution"]["correct"]
            p = m["Correctness Distribution"]["partial"]
            i = m["Correctness Distribution"]["incorrect"]
            model_correctness_data[model] = [c, p, i]
        
        base_colors = ["#1E90FF", "#E67E22", "#9B59B6", "#2ECC71", "#F1C40F", "#E74C3C", "#607D8B", "#2C3E50"]
        model_colors = {model: base_colors[i % len(base_colors)] for i, model in enumerate(models)}
        
        # Set Matplotlib parameters for dark theme
        plt.rcParams['figure.facecolor'] = 'black'
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['axes.edgecolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['axes.titlecolor'] = 'white'
        plt.rcParams['legend.facecolor'] = 'black'
        plt.rcParams['legend.edgecolor'] = 'white'
        plt.rcParams['legend.labelcolor'] = 'white'
        plt.rcParams['font.family'] = 'Noto Sans Arabic'
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor='black')
        self.fig = fig
        
        # --- Left Subplot: Main Metrics ---
        ax1 = axes[0]
        x_metrics = np.arange(len(metric_names))
        bar_width = 0.5 / len(models)
        all_vals = [val for vals in model_metric_data.values() for val in vals]
        global_max = max(all_vals) if all_vals else 1
        ax1.set_ylim(0, global_max * 1.25)
        for i, model in enumerate(models):
            vals = model_metric_data[model]
            offset = i * bar_width * 1.2
            ax1.bar(x_metrics + offset, vals, bar_width, color=model_colors[model],
                    edgecolor='white', linewidth=1.5)
            for j, val in enumerate(vals):
                ax1.text(j + offset, val + global_max * 0.03, f"{val:.2f}",
                         ha='center', va='bottom', color='white',
                         fontsize=12, fontweight='bold')  # <--- bigger font
        ax1.set_title(ar("Metrics by Model / مقاييس حسب النموذج"), fontsize=14, color='white', pad=15)
        ax1.set_xticks(x_metrics + bar_width*(len(models)-1)/2)
        ax1.set_xticklabels(metric_names, rotation=30, ha='right', fontsize=11, color='white')
        ax1.set_ylabel(ar("Value / Percentage / القيمة أو النسبة"), fontsize=12, color='white')
        ax1.grid(axis='y', linestyle='--', alpha=0.4, color='white')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('white')
        ax1.spines['bottom'].set_color('white')
        legend_patches = [mpatches.Patch(color=model_colors[m], label=m) for m in models]
        ax1.legend(handles=legend_patches, fontsize=10, frameon=False, loc='upper left')
        
        # --- Middle Subplot: Sentiment Distribution ---
        ax2 = axes[1]
        x_models = np.arange(len(models))
        bar_width2 = 0.2
        all_sent_vals = [val for vals in model_sentiment_data.values() for val in vals]
        max_sent = max(all_sent_vals) if all_sent_vals else 100
        ax2.set_ylim(0, max_sent * 1.25)
        sentiment_colors = {
            ar("Positive / إيجابي"): "green",
            ar("Neutral / محايد"): "gray",
            ar("Negative / سلبي"): "red"
        }
        for i, sentiment in enumerate(sentiments):
            offset = (i - 1) * bar_width2
            vals = [model_sentiment_data[m][i] for m in models]
            ax2.bar(x_models + offset, vals, bar_width2, color=sentiment_colors[sentiment],
                    edgecolor='white', linewidth=1.5)
            for j, val in enumerate(vals):
                ax2.text(j + offset, val + max_sent * 0.03, f"{val:.1f}%",
                         ha='center', va='bottom', color='white',
                         fontsize=12, fontweight='bold')  # <--- bigger font
        ax2.set_title(ar("Sentiment Distribution / توزيع المشاعر"), fontsize=14, color='white', pad=15)
        ax2.set_xticks(x_models)
        ax2.set_xticklabels(models, rotation=30, ha='right', fontsize=11, color='white')
        ax2.set_ylabel(ar("Percentage (%) / النسبة المئوية"), fontsize=12, color='white')
        ax2.grid(axis='y', linestyle='--', alpha=0.4, color='white')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('white')
        ax2.spines['bottom'].set_color('white')
        sentiment_patches = [
            mpatches.Patch(color='green', label=ar("Positive / إيجابي")),
            mpatches.Patch(color='gray', label=ar("Neutral / محايد")),
            mpatches.Patch(color='red', label=ar("Negative / سلبي"))
        ]
        ax2.legend(handles=sentiment_patches, fontsize=10, frameon=False, loc='upper left')
        
        # --- Right Subplot: Correctness Distribution ---
        ax3 = axes[2]
        bar_width3 = 0.2
        correctness_labels = ["correct", "partial", "incorrect"]
        correctness_colors = {
            "correct": "#2ECC71",   # green
            "partial": "#F1C40F",   # yellow
            "incorrect": "#E74C3C"  # red
        }
        for i, label in enumerate(correctness_labels):
            offset = (i - 1) * bar_width3
            vals = [model_correctness_data[m][i] for m in models]
            ax3.bar(x_models + offset, vals, bar_width3,
                    color=correctness_colors[label],
                    edgecolor='white', linewidth=1.5,
                    label=label.capitalize())
            for j, val in enumerate(vals):
                ax3.text(j + offset, val + 2, f"{val:.1f}%",
                         ha='center', va='bottom', color='white',
                         fontsize=12, fontweight='bold')  # <--- bigger font
        ax3.set_title(ar("Correctness Distribution / توزيع الصحة"), fontsize=14, color='white', pad=15)
        ax3.set_xticks(x_models)
        ax3.set_xticklabels(models, rotation=30, ha='right', fontsize=11, color='white')
        ax3.set_ylabel(ar("Percentage (%) / النسبة المئوية"), fontsize=12, color='white')
        ax3.grid(axis='y', linestyle='--', alpha=0.4, color='white')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_color('white')
        ax3.spines['bottom'].set_color('white')
        ax3.legend(fontsize=10, frameon=False, loc='upper left')
        
        plt.tight_layout()
        plt.show()
        messagebox.showinfo("Comparison Complete", "Multi-benchmark comparison displayed.\nUse 'Save Graph' to save the figure.")

    def save_graph(self):
        if self.fig is None:
            messagebox.showerror("Error", "No figure to save. Please run 'Compare Results' first.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"), ("All Files", "*.*")])
        if filename:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Save Successful", f"Graph saved to: {filename}")

    def quit_app(self):
        if messagebox.askyesno("Quit", "Are you sure you want to quit?"):
            self.root.destroy()

if __name__ == "__main__":
    try:
        import textblob
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "textblob"])
    root = tk.Tk()
    app = ZBenchGUI(root)
    root.mainloop()
