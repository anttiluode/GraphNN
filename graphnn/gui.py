"""Tk interface. Workers never touch a widget or a Tk variable."""
import argparse
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk
import torch

from .data import demo_pair, image_from_array, load_image
from .training import PairSession


BG, PANEL, TEXT, MUTED, ACCENT = "#0b1220", "#141e30", "#edf3ff", "#95a8c5", "#43dac4"


class JanusApp:
    def __init__(self, root, session=None):
        self.root = root
        self.session = session or PairSession()
        self.closed = False
        self.busy = False
        self.worker = None
        self.stop_event = threading.Event()
        self.messages = queue.Queue()
        self._render_active = False
        self._render_dirty = False
        self._generation = 0
        self.display_images = {}
        self.display_arrays = {}
        self.display_angle = 0.
        self.image_slots = [None,None]
        self.buttons = []
        self.angle = tk.DoubleVar(root,0.)
        self.control = tk.StringVar(root,"stored")
        self.channel = tk.IntVar(root,0)
        self.step_limit = tk.StringVar(root,"1000")
        self.query_text = tk.StringVar(root)
        self.status = tk.StringVar(root,"Ready. Load images or try the generated pair.")
        self.metric_text = tk.StringVar(root)
        self.root.title("Janus Graph — GraphNN")
        self.root.geometry("1220x850")
        self.root.minsize(1080,780)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW",self.close)
        self._build()
        if self.session.images is None and not self.session.memories:
            self.session.set_pair(demo_pair(self.session.config.image_size,41))
        if self.session.images is not None:
            self.image_slots = list(self.session.images.copy())
        self._thumbnails()
        self.angle.trace_add("write",lambda *_:self.request_render())
        self.control.trace_add("write",lambda *_:self.request_render())
        self.channel.trace_add("write",lambda *_:self.request_render())
        self._after_id = self.root.after(30,self._poll)
        self.request_render()

    def _label(self, parent, text=None, variable=None, size=11, color=TEXT, **kwargs):
        return tk.Label(parent,text=text,textvariable=variable,font=("Segoe UI",size),
                        bg=parent.cget("bg"),fg=color,**kwargs)

    def _button(self, parent, text, command, primary=False):
        button = tk.Button(parent,text=text,command=command,font=("Segoe UI",10,"bold"),
                           bg=ACCENT if primary else "#25344d",fg=BG if primary else TEXT,
                           activebackground="#5ee8d3",activeforeground=BG,relief="flat",bd=0,
                           padx=12,pady=9,cursor="hand2",disabledforeground=MUTED)
        self.buttons.append(button)
        return button

    def _build(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TCombobox",fieldbackground=PANEL,background="#25344d",foreground=TEXT)
        header = tk.Frame(self.root,bg=BG)
        header.pack(fill="x",padx=24,pady=(16,12))
        self._label(header,"JANUS / GRAPHNN",size=23).pack(anchor="w")
        self._label(header,"Two images. One stored graph. A question chooses the answer.",color=MUTED).pack(anchor="w",pady=(4,0))
        body = tk.Frame(self.root,bg=BG)
        body.pack(fill="both",expand=True,padx=24)
        left = tk.Frame(body,bg=PANEL,width=235)
        left.pack(side="left",fill="y",padx=(0,18))
        left.pack_propagate(False)
        self._label(left,"SOURCE IMAGES",size=10,color=MUTED).pack(anchor="w",padx=14,pady=(14,8))
        self.target_labels = []
        for slot,name in enumerate(("A · query 0°","B · query 90°")):
            self._button(left,f"Load image {name[0]}",lambda s=slot:self.load_source(s)).pack(fill="x",padx=14,pady=(0,5))
            label = tk.Label(left,bg="#090f1a",fg=MUTED,text=name)
            label.pack(padx=14,pady=(0,7))
            self.target_labels.append(label)
        self._button(left,"Generated demo pair",self.demo).pack(fill="x",padx=14,pady=5)
        self._label(left,"Additional fit steps",color=MUTED,size=10).pack(anchor="w",padx=14,pady=(12,3))
        tk.Entry(left,textvariable=self.step_limit,bg="#25344d",fg=TEXT,insertbackground=TEXT,relief="flat").pack(fill="x",padx=14,ipady=6)
        self.train_button = self._button(left,"Fit loaded pair",self.toggle_train,True)
        self.train_button.pack(fill="x",padx=14,pady=(8,5))
        self._button(left,"Encode with frozen weights",self.encode).pack(fill="x",padx=14,pady=5)
        self._button(left,"Load saved memory",self.load_checkpoint).pack(fill="x",padx=14,pady=5)
        self._button(left,"Save memory",self.save_checkpoint).pack(fill="x",padx=14,pady=5)
        self._button(left,"Export graph view",self.export).pack(fill="x",padx=14,pady=5)
        self._button(left,"Reset network",self.reset).pack(fill="x",padx=14,pady=5)
        right = tk.Frame(body,bg=BG)
        right.pack(side="left",fill="both",expand=True)
        outputs = tk.Frame(right,bg=BG)
        outputs.pack(fill="x")
        self.output_labels = {}
        for arm,title in (("graph","GRAPH RESPONSE"),("vector","DIRECT / VECTOR CONTROL")):
            card = tk.Frame(outputs,bg=PANEL)
            card.pack(side="left",fill="both",expand=True,padx=(0,10) if arm=="graph" else (0,0))
            self._label(card,title,size=11,color=ACCENT if arm=="graph" else MUTED).pack(anchor="w",padx=14,pady=(12,8))
            label = tk.Label(card,bg="#090f1a",width=320,height=320)
            label.pack(padx=12,pady=(0,12))
            self.output_labels[arm] = label
        query = tk.Frame(right,bg=PANEL)
        query.pack(fill="x",pady=(12,0))
        self._label(query,variable=self.query_text,size=12).pack(anchor="w",padx=14,pady=(10,0))
        tk.Scale(query,from_=0,to=90,resolution=1,orient="horizontal",variable=self.angle,
                 bg=PANEL,fg=TEXT,troughcolor="#25344d",activebackground=ACCENT,
                 highlightthickness=0,sliderlength=24).pack(fill="x",padx=12,pady=(0,6))
        controls = tk.Frame(query,bg=PANEL)
        controls.pack(fill="x",padx=14,pady=(0,12))
        self._label(controls,"Read memory:",color=MUTED).pack(side="left")
        ttk.Combobox(controls,textvariable=self.control,values=("stored","randomized","blank"),state="readonly",width=12).pack(side="left",padx=8)
        self._label(controls,"Map channel:",color=MUTED).pack(side="left",padx=(12,4))
        self.channel_box = ttk.Combobox(controls,textvariable=self.channel,values=list(range(self.session.config.channels)),state="readonly",width=4)
        self.channel_box.pack(side="left")
        memory = tk.Frame(right,bg=PANEL)
        memory.pack(fill="both",expand=True,pady=(12,0))
        self.map_labels = {}
        for name,title in (("gain","INCIDENT GAIN"),("phase","PHASE BYTE")):
            frame = tk.Frame(memory,bg=PANEL)
            frame.pack(side="left",padx=12,pady=10)
            self._label(frame,title,color=MUTED,size=9).pack(anchor="w",pady=(0,6))
            label = tk.Label(frame,bg="#090f1a")
            label.pack()
            self.map_labels[name] = label
        self._label(memory,variable=self.metric_text,justify="left",anchor="nw",color=MUTED,size=10).pack(side="left",fill="both",expand=True,padx=8,pady=12)
        self._label(self.root,variable=self.status,anchor="w",color=MUTED,size=10).pack(fill="x",padx=24,pady=12)

    def _thumbnails(self):
        for slot,label in enumerate(self.target_labels):
            image = self.image_slots[slot]
            if image is None:
                label.configure(image="",text="No source pixels loaded",width=26,height=3)
            else:
                photo = ImageTk.PhotoImage(image_from_array(image).resize((120,88),Image.Resampling.NEAREST))
                self.display_images[f"target{slot}"] = photo
                label.configure(image=photo,text="",width=120,height=88)

    def _set_busy(self, value, kind=""):
        self.busy = value
        for button in self.buttons:
            button.configure(state="disabled" if value else "normal")
        if value and kind == "fit":
            self.train_button.configure(state="normal",text="Pause fitting")
        else:
            self.train_button.configure(text="Fit loaded pair",state="normal" if not value and self.session.images is not None else "disabled")

    def _job(self, kind, function):
        if self.busy or self.closed:
            return
        self._generation += 1
        self.stop_event.clear()
        self._set_busy(True,kind)
        self.status.set({"fit":"Fitting this pair; shared weights are changing.","encode":"Encoding with frozen shared weights.","load":"Loading saved memory...","save":"Saving memory...","reset":"Resetting..."}.get(kind,"Working..."))
        def worker():
            try:
                function()
                self.messages.put(("done",kind))
            except Exception as error:
                self.messages.put(("error",str(error)))
        self.worker = threading.Thread(target=worker,daemon=True)
        self.worker.start()

    def toggle_train(self):
        if self.busy:
            self.stop_event.set()
            self.train_button.configure(state="disabled",text="Pausing...")
            return
        try:
            steps = int(self.step_limit.get())
            if not 1 <= steps <= 100000:
                raise ValueError("Choose 1..100000 additional steps")
        except ValueError as error:
            messagebox.showerror("Fit steps",str(error),parent=self.root)
            return
        def fit():
            for step in range(steps):
                if self.stop_event.is_set():
                    break
                self.session.step()
                if step % 10 == 0:
                    self.messages.put(("progress",self.session.steps))
                self.stop_event.wait(.002)
        self._job("fit",fit)

    def request_render(self):
        if self.closed:
            return
        angle = self.angle.get()
        self.query_text.set(f"Query {angle:.0f}° · " + ("image A" if angle==0 else "image B" if angle==90 else "intermediate query · no supervised target"))
        self._render_dirty = True
        if self._render_active or not self.session.memories:
            return
        self._render_dirty = False
        self._render_active = True
        generation,control,channel = self._generation,self.control.get(),self.channel.get()
        def render():
            try:
                with self.session.lock:
                    result = (self.session.render(angle,control),self.session.maps(channel),self.session.metrics(),self.session.steps,self.session.mode)
                self.messages.put(("render",generation,angle,result))
            except Exception as error:
                self.messages.put(("render_error",str(error)))
        threading.Thread(target=render,daemon=True).start()

    def _show(self, angle, result):
        answers,maps,metrics,steps,mode = result
        self.display_arrays = answers
        self.display_angle = angle
        for arm,array in answers.items():
            photo = ImageTk.PhotoImage(image_from_array(array).resize((320,320),Image.Resampling.NEAREST))
            self.display_images[arm] = photo
            self.output_labels[arm].configure(image=photo,width=320,height=320)
        gain,phase = maps
        gain_rgb = np.stack((gain*.3,gain,gain*.85),axis=-1)
        hsv = np.stack((phase*255,np.full_like(phase,190),np.full_like(phase,225)),axis=-1).astype(np.uint8)
        images = (image_from_array(gain_rgb),Image.fromarray(hsv,"HSV").convert("RGB"))
        for name,image in zip(("gain","phase"),images):
            photo = ImageTk.PhotoImage(image.resize((120,120),Image.Resampling.NEAREST))
            self.display_images[name] = photo
            self.map_labels[name].configure(image=photo)
        lines = [f"{metrics['graph']['payload_bytes']:,} stored bytes per pair / arm",
                 f"{metrics['graph']['shared_parameters']:,} shared parameters / arm",
                 f"{steps:,} steps · {self.session.device}",f"Model: {mode}",""]
        for arm in ("graph","vector"):
            metric = metrics[arm]
            lines.append(f"{arm.title()} endpoints: MSE {metric['mse']:.5f} · {metric['psnr_db']:.2f} dB" if "mse" in metric else f"{arm.title()}: saved memory; targets unavailable")
        lines.extend(("","Metrics use stored memory at 0° / 90°.","Map shows stored edges; control affects reads."))
        self.metric_text.set("\n".join(lines))

    def _poll(self):
        if self.closed:
            return
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            kind = message[0]
            if kind == "render":
                self._render_active = False
                if message[1] == self._generation:
                    self._show(message[2],message[3])
            elif kind == "render_error":
                self._render_active = False
                self.status.set(message[1])
            elif kind == "progress":
                self.status.set(f"Fitting loaded pair · step {message[1]:,}. Move the slider to inspect.")
                self.request_render()
            elif kind == "done":
                self._set_busy(False)
                if message[1] == "load":
                    self.image_slots = [None,None]
                    self.channel.set(0)
                    self.channel_box.configure(values=list(range(self.session.config.channels)))
                    self._thumbnails()
                self.status.set("Saved memory loaded. Reads need no source pixels." if message[1]=="load" else "Ready. Shared weights stay fixed until you fit again.")
                self.request_render()
            elif kind == "error":
                self._set_busy(False)
                self.status.set(message[1])
                messagebox.showerror("GraphNN",message[1],parent=self.root)
        if self._render_dirty and not self._render_active:
            self.request_render()
        self._after_id = self.root.after(30,self._poll)

    def load_source(self, slot, path=None):
        if self.busy:
            return
        path = path or filedialog.askopenfilename(parent=self.root,title=f"Load image {'AB'[slot]}",filetypes=(("Images","*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),("All files","*")))
        if not path:
            return
        try:
            self.image_slots[slot] = load_image(path,self.session.config.image_size)
            self._thumbnails()
            if all(i is not None for i in self.image_slots):
                pair = np.stack(self.image_slots)
                self._job("encode",lambda:self.session.set_pair(pair))
            else:
                self.status.set("Load the other image to encode a complete pair with frozen weights.")
        except Exception as error:
            messagebox.showerror("Image",str(error),parent=self.root)

    def demo(self):
        pair = demo_pair(self.session.config.image_size,41+self.session.steps)
        self.image_slots = list(pair)
        self._thumbnails()
        self._job("encode",lambda:self.session.set_pair(pair))

    def encode(self):
        self._job("encode",self.session.encode)

    def reset(self):
        if self.busy:
            return
        pair = np.stack(self.image_slots) if all(i is not None for i in self.image_slots) else demo_pair(self.session.config.image_size,41)
        self.session = PairSession(self.session.config,str(self.session.device))
        self.session.set_pair(pair)
        self.image_slots = list(pair)
        self._generation += 1
        self._thumbnails()
        self._set_busy(False)
        self.request_render()

    def load_checkpoint(self, path=None):
        path = path or filedialog.askopenfilename(parent=self.root,title="Load GraphNN memory",filetypes=(("GraphNN memory","*.pt"),))
        if path:
            self._job("load",lambda:self.session.load(path))

    def save_checkpoint(self):
        path = filedialog.asksaveasfilename(parent=self.root,title="Save GraphNN memory",defaultextension=".pt",filetypes=(("GraphNN memory","*.pt"),))
        if path:
            self._job("save",lambda:self.session.save(path))

    def export(self):
        if "graph" not in self.display_arrays:
            return
        path = filedialog.asksaveasfilename(parent=self.root,title="Export displayed graph view",defaultextension=".png",initialfile=f"graph_query_{self.display_angle:.0f}.png",filetypes=(("PNG","*.png"),))
        if path:
            try:
                image_from_array(self.display_arrays["graph"]).save(path)
                self.status.set(f"Exported the displayed {self.display_angle:.0f}° graph view.")
            except OSError as error:
                messagebox.showerror("Export",str(error),parent=self.root)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        self.root.after_cancel(self._after_id)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Janus Graph: load images, write memory, query with a slider")
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--checkpoint",type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        session = PairSession(device=args.device)
        if args.checkpoint:
            session.load(args.checkpoint)
        else:
            bundled = Path(__file__).resolve().parents[1]/"results"/"pilot"/"memory.pt"
            if bundled.exists():
                session.load(bundled)
                session.set_pair(demo_pair(session.config.image_size,41))
        root = tk.Tk()
    except (ValueError,tk.TclError) as error:
        parser.exit(1,f"Cannot start Janus Graph: {error}\nLinux: install python3-tk and use a desktop display.\n")
    app = JanusApp(root,session)
    root.mainloop()
    if app.worker is not None:
        app.worker.join(timeout=5)
