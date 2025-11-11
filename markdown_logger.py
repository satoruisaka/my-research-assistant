# markdown_logger.py
"""
Records log sessions in markdown format
"""

from datetime import datetime
import os

class MarkdownLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(log_dir, f"MRAlog_{timestamp}.md")
        self.file = open(self.log_path, "w", encoding="utf-8")
        self.file.write(f"# Session Log — {timestamp}\n\n---\n")
        self.turn = 1

    def log_turn(self, task_type, user_query, prompt, response, citations=None, top_chunks=None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"\n## Turn {self.turn}\n\n")
        self.file.write(f"**Timestamp:** {ts}  \n")
        self.file.write(f"**Task:** {task_type}  \n")
        self.file.write(f"**User Query:** {user_query}  \n\n")
        self.file.write(f"**Prompt:**  \n{prompt}\n\n")
        self.file.write(f"**Response:**  \n{response}\n\n")
        if citations:
            self.file.write("**Citations:**  \n" + citations + "\n")
#        if top_chunks:
#            self.file.write("**Top Chunks Retrieved:**  \n")
#            for chunk in top_chunks:
#                self.file.write(f"- {chunk}\n")
        self.file.write("\n---\n")
        self.file.flush()
        self.turn += 1

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()