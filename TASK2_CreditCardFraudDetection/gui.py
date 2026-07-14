# Credit Card Fraud Detection System GUI
import pandas as pd
import tkinter as tk
from tkinter import ttk

# Load predictions
df = pd.read_csv("fraud_predictions.csv")

# Filter risky or fraud transactions
df_risky = df[df["Final_Alert"].str.contains("HIGH RISK|FRAUD", case=False)]


root = tk.Tk()
root.title("Credit Card Fraud Detection System")
root.geometry("1980x1080+0+0")
root.state("zoomed")
background_color = "#12012d"
root.configure(bg=background_color)


label = tk.Label(root, text="High Risk / Fraud Transactions", font=("Arial", 14, "bold"), bg=background_color, fg="white")
label.pack(pady=10)

frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

#scrollbars
scroll_y = tk.Scrollbar(frame)
scroll_y.pack(side="right", fill="y")

scroll_x = tk.Scrollbar(frame, orient="horizontal")
scroll_x.pack(side="bottom", fill="x")


columns = list(df_risky.columns)

tree = ttk.Treeview(frame,
                    columns=columns,
                    show="headings",
                    yscrollcommand=scroll_y.set,
                    xscrollcommand=scroll_x.set)

scroll_y.config(command=tree.yview)
scroll_x.config(command=tree.xview)


for col in columns:
    tree.heading(col, text=col)
    tree.column(col, width=120)


for _, row in df_risky.iterrows():
    tree.insert("", "end", values=list(row))

tree.pack(fill="both", expand=True)

# Run
root.mainloop()