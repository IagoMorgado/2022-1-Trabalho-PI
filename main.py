

import threading
import tkinter as tk
from _thread import *
from tkinter import Menu, Frame, Label, HORIZONTAL, TOP, X, YES
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askquestion, showerror, showinfo
from tkinter.ttk import Progressbar

import numpy
from PIL import Image, ImageTk, ImageOps

from algorithm import trainclf, processImageAndComputeDescriptors, \
    saveclf, loadclf

scale_constant = 1.2

selection_rect_offset = 32

min_width = 500
min_height = 500


class TrainWindow(tk.Toplevel):
    def __init__(self, root):
        super().__init__(root)
        self.geometry("300x200")
        self.mainframe = Frame(self)
        self.labelVar = tk.StringVar()
        self.label = Label(self.mainframe, textvariable=self.labelVar)
        self.label.pack(side=TOP, fill=X, expand=YES)
        self.progress = Progressbar(self.mainframe, orient=HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(side=TOP, fill=X, expand=YES)
        self.mainframe.pack()


class SubWindow(tk.Toplevel):
    def __init__(self, image, root, clf):
        super().__init__(root)
        self.geometry("300x200")
        self.image_original = image
        self.image_on_screen = self.image_original
        self.num_of_colors = 256
        self.resolution = 128
        self.should_equalize_image = False
        self.clf = clf

        self.cropped_image_menu = Menu(self)
        self.config(menu=self.cropped_image_menu)

        self.resolution_submenu = Menu(self.cropped_image_menu, tearoff=0)
        self.cropped_image_menu.add_cascade(label='Resolução', menu=self.resolution_submenu)
        self.resolution_submenu.add_command(label='128 x 128', command=lambda: self.change_resolution(128))
        self.resolution_submenu.add_command(label='64 x 64', command=lambda: self.change_resolution(64))
        self.resolution_submenu.add_command(label='32 x 32', command=lambda: self.change_resolution(32))
        self.resolution_submenu.add_command(label='16 x 16', command=lambda: self.change_resolution(16))
        self.resolution_submenu.add_command(label='8 x 8', command=lambda: self.change_resolution(8))
        self.resolution_submenu.add_command(label='4 x 4', command=lambda: self.change_resolution(4))
        self.resolution_submenu.add_command(label='2 x 2', command=lambda: self.change_resolution(2))

        self.quantize_submenu = Menu(self.cropped_image_menu, tearoff=0)
        self.cropped_image_menu.add_cascade(label='Quantização', menu=self.quantize_submenu)
        self.quantize_submenu.add_command(label='32', command=lambda: self.change_quantize(32))
        self.quantize_submenu.add_command(label='16', command=lambda: self.change_quantize(16))
        self.quantize_submenu.add_command(label='8', command=lambda: self.change_quantize(8))

        self.cropped_image_menu.add_command(label='Equalizar', command=self.equalize_image)

        self.cropped_image_menu.add_command(label='Reset', command=self.reset)

        self.cropped_image_menu.add_command(label='Classificar região', command=self.classify_image)

        width, height = self.image_original.size

        self.canvas = tk.Canvas(self, width=width, height=height)

        self.photo_image = ImageTk.PhotoImage(self.image_on_screen)
        self.image_canvas = self.canvas.create_image(0, 0, image=self.photo_image, anchor='nw')
        self.canvas.pack()

    def classify_image(self):
        if self.clf is None:
            print("CLF IS NONE")
            showerror("Error", "Nenhum classificador treinado foi selecionado")
            return
        if self.image_original is None:
            print("IMAGE IS NONE")
            showerror("Error", "Não há nenhuma imagem para ser classificada")
            return
        descriptors = processImageAndComputeDescriptors(image=self.image_on_screen)
        descriptors = numpy.reshape(descriptors, (1, -1))
        predicted = self.clf.predict(descriptors)
        showinfo("Classificação", f"Birads {predicted[0]}")
        print(predicted)

    def change_quantize(self, colors):
        self.num_of_colors = colors
        self.re_draw()

    def change_resolution(self, resolution):
        self.resolution = resolution
        self.re_draw()

    def equalize_image(self):
        self.should_equalize_image = not self.should_equalize_image
        self.re_draw()

    def re_draw(self):
        self.image_on_screen = self.image_original.quantize(colors=self.num_of_colors)
        self.image_on_screen = self.image_on_screen.resize((self.resolution, self.resolution))
        self.image_on_screen = self.image_on_screen.resize((128, 128))

        if self.should_equalize_image:
            self.image_on_screen = ImageOps.equalize(self.image_on_screen)

        self.photo_image = ImageTk.PhotoImage(self.image_on_screen)
        self.image_canvas = self.canvas.create_image(0, 0, image=self.photo_image, anchor='nw')

    def reset(self):
        self.num_of_colors = 256
        self.resolution = 128
        self.should_equalize_image = False
        self.re_draw()

    def set_image(self, image):
        self.image_original = image
        self.re_draw()


class MainWindow:
    def __init__(self):
        # Screen
        self.window = tk.Tk()
        self.window.eval('tk::PlaceWindow . center')
        self.window.geometry("500x500")
        self.window.title("Trabalho Prático - Processamento de Imagens")
        self.filepath = ""
        self.image_original = None
        self.photo_image = None
        self.scale = 1
        self.selection_enabled = False
        self.selection_rect = None
        self.exist_sub_window = None
        self.image_canvas = None
        self.clf = None

        self.menu = Menu(self.window)
        self.window.config(menu=self.menu)
        self.print_lock = threading.Lock()
        self.print_lock.acquire()
        self.menu.add_command(label='Abrir Imagem', command=self.open_image)
        self.menu.add_command(label="Selecionar Área", command=self.selection_area)
        self.menu.add_command(label="Treinar classificador", command=self.train)
        self.menu.add_command(label="Salvar classificador", command=self.save_train)
        self.menu.add_command(label="Carregar classificador", command=self.load_train)
        self.menu.add_command(label="Classificar imagem", command=self.classify_image)
        self.image_on_screen = None

        self.canvas = tk.Canvas(self.window, width=800, height=800)
        self.canvas.pack()

        self.canvas.bind('<Button-1>', self.on_click)
        self.window.mainloop()

    def classify_image(self):
        if self.clf is None:
            print("CLF IS NONE")
            showerror("Error", "Não há nenhum classificador treinado")
            return
        if self.image_original is None:
            print("IMAGE IS NONE")
            showerror("Error", "Não há nenhuma imagem para ser classificada")
            return
        descriptors = processImageAndComputeDescriptors(image=self.image_original)
        descriptors = numpy.reshape(descriptors, (1, -1))
        predicted = self.clf.predict(descriptors)
        showinfo("Classificação", f"Birads {predicted[0]}")
        print(predicted)

    def save_train(self):
        if self.clf is not None:
            should_save_train = askquestion("Trabalho PI", message="Deseja salvar o classificador teinado ?") == "yes"
            if should_save_train:
                saveclf(self.clf)
        else:
            showerror("Error", "Não há nenhum classificador treinado, não foi possível salvar")

    def load_train(self):
        should_load_train = askquestion("Trabalho PI", message="Deseja carregar o classificador salvo ?") == "yes"
        if should_load_train:
            self.clf = loadclf()

    def get_clf(self, train_window):
        self.clf = trainclf(train_window)

    def train(self):
        train_window = TrainWindow(self.window)
        start_new_thread(self.get_clf, (train_window,))

    def selection_area(self):
        self.selection_enabled = not self.selection_enabled

    def open_cropped_image_window(self, image):
        if self.should_create_sub_window():
            self.exist_sub_window = SubWindow(image, self.window, self.clf)
        else:
            self.exist_sub_window.set_image(image)
            self.exist_sub_window.clf = self.clf

    def should_create_sub_window(self):
        return self.exist_sub_window is None or not self.exist_sub_window.winfo_exists()

    def on_click(self, event):
        self.canvas.scan_mark(event.x, event.y)

        if self.selection_enabled:
            if self.selection_rect is not None:
                self.canvas.delete(self.selection_rect)

            self.draw_selection_rectangle(event.x, event.y)
            crop_limits = self.get_crop_area_limits(event.x, event.y)
            cropped_image = self.image_on_screen.crop(crop_limits)

            self.open_cropped_image_window(cropped_image)

    def get_crop_area_limits(self, x_window, y_window):
        x_min = x_window - selection_rect_offset
        y_min = y_window - selection_rect_offset
        x_max = x_window + selection_rect_offset
        y_max = y_window + selection_rect_offset
        return x_min, y_min, x_max, y_max

    def get_selection_rectangle_limits(self, x_window, y_window):
        x_center, y_center = self.canvas.canvasx(x_window), self.canvas.canvasy(y_window)
        x_min, y_min = x_center - selection_rect_offset, y_center - selection_rect_offset
        x_max, y_max = x_center + selection_rect_offset, y_center + selection_rect_offset
        return x_max, x_min, y_max, y_min

    def draw_selection_rectangle(self, x_window, y_window):
        x_max, x_min, y_max, y_min = self.get_selection_rectangle_limits(x_window, y_window)
        self.selection_rect = self.canvas.create_rectangle(x_min, y_min, x_max, y_max, dash=(4, 1), outline="blue")

    def open_image(self):
        self.filepath = askopenfilename(
            filetypes=[("Image Files", "*.png *.tif")]
        )
        if self.filepath == '':
            return
        self.image_original = (Image.open(self.filepath)).convert("L")
        self.image_on_screen = self.image_original
        self.photo_image = ImageTk.PhotoImage(self.image_on_screen)
        self.image_canvas = self.canvas.create_image(0, 0, image=self.photo_image, anchor='nw')
        width, height = self.image_on_screen.size
        if width < min_width:
            width = min_width
        if height < min_height:
            height = min_height
        self.window.geometry(f"{width}x{height}")
        self.canvas.config(width=width, height=height)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


def main():
    MainWindow()


if __name__ == '__main__':
    main()
