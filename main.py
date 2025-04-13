import sys
import json
import serial
import serial.tools.list_ports
import folium
import io
import time
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QVBoxLayout, QLabel
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWebEngineWidgets import QWebEngineView
from window_ui import Ui_MainWindow
import sqlite3
from datetime import datetime
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtGui import QPixmap, QTransform, QPainter
import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import math
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
import os
from collections import Counter

MODEL_SAVE_PATH = 'transformer_repair_model.pth'
VOCAB_SAVE_PATH = 'vocab.json'
TOKENIZER_SAVE_PATH = 'tokenizer_config.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, SEP_IDX = 0, 1, 2, 3, 4

special_symbols = ['<pad>', '<sos>', '<eos>', '<unk>', '<sep>']

MAX_SEQ_LEN = 512

class CharacterTokenizer:
    def __init__(self, special_symbols):
        self.char2idx = {}
        self.idx2char = {}
        self.special_symbols = special_symbols
        self._build_vocab([]) # Инициализация со спец. символами

    def _build_vocab(self, texts):
        all_chars = Counter()
        for text in texts:
            all_chars.update(text)

        self.char2idx = {char: i for i, char in enumerate(special_symbols)}
        for char, _ in all_chars.most_common():
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)
        self.idx2char = {i: char for char, i in self.char2idx.items()}

        # Обновляем глобальные переменные размера словаря
        global SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
        SRC_VOCAB_SIZE = len(self.char2idx)
        TGT_VOCAB_SIZE = SRC_VOCAB_SIZE
        print(f"Vocabulary size: {SRC_VOCAB_SIZE}")
        print(f"Sample vocab mapping: {list(self.char2idx.items())[:10]}...")

    def encode(self, text, add_special_tokens=True):
        tokens = [self.char2idx.get(char, UNK_IDX) for char in text]
        if add_special_tokens:
            return [SOS_IDX] + tokens + [EOS_IDX]
        return tokens

    def decode(self, token_ids, remove_special_tokens=True):
        chars = []
        for token_id in token_ids:
            if remove_special_tokens and token_id in [PAD_IDX, SOS_IDX, EOS_IDX, SEP_IDX]:
                continue
            chars.append(self.idx2char.get(token_id, '<unk>')) # Используем <unk> для неизвестных ID
        return "".join(chars)

    def save(self, vocab_path, config_path):
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({'char2idx': self.char2idx, 'idx2char': self.idx2char}, f, ensure_ascii=False, indent=4)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({'special_symbols': self.special_symbols,
                       'pad_idx': PAD_IDX, 'sos_idx': SOS_IDX, 'eos_idx': EOS_IDX,
                       'unk_idx': UNK_IDX, 'sep_idx': SEP_IDX}, f, indent=4)

    @classmethod
    def load(cls, vocab_path, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        instance = cls(config['special_symbols'])

        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocabs = json.load(f)
            # JSON ключи словарей могут быть строками, преобразуем обратно в int для idx2char
            instance.char2idx = vocabs['char2idx']
            instance.idx2char = {int(k): v for k, v in vocabs['idx2char'].items()}

        # Обновляем глобальные переменные размера словаря
        global SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
        SRC_VOCAB_SIZE = len(instance.char2idx)
        TGT_VOCAB_SIZE = SRC_VOCAB_SIZE

        # Убедимся, что глобальные IDX константы соответствуют загруженным
        global PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, SEP_IDX
        PAD_IDX = config['pad_idx']
        SOS_IDX = config['sos_idx']
        EOS_IDX = config['eos_idx']
        UNK_IDX = config['unk_idx']
        SEP_IDX = config['sep_idx']

        print(f"Loaded tokenizer. Vocabulary size: {SRC_VOCAB_SIZE}")
        return instance


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0) # Добавим batch dimension в начале

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding) # Не параметр модели

    def forward(self, token_embedding: torch.Tensor):
        # token_embedding: [batch_size, seq_len, emb_size]
        # self.pos_embedding[:, :token_embedding.size(1)] -> [1, seq_len, emb_size]
        seq_len = token_embedding.size(1)
        pos_emb = self.pos_embedding[:, :seq_len]
        # Убедимся, что размер совпадает перед сложением
        # print(f"Token emb shape: {token_embedding.shape}")
        # print(f"Positional emb shape: {pos_emb.shape}")
        if pos_emb.shape[1] != seq_len:
            # Этого не должно происходить при правильном maxlen, но на всякий случай
            raise ValueError(f"Positional embedding sequence length ({pos_emb.shape[1]}) does not match input sequence length ({seq_len}). Increase maxlen in PositionalEncoding.")

        return self.dropout(token_embedding + pos_emb)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = MAX_SEQ_LEN): # max_seq_len для PositionalEncoding
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True) # Важно: batch_first=True
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, maxlen=max_seq_len) # Используем max_seq_len

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                src_padding_mask: torch.Tensor,
                tgt_padding_mask: torch.Tensor,
                memory_key_padding_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        # src: [batch_size, src_seq_len]
        # trg: [batch_size, tgt_seq_len]
        # src_padding_mask: [batch_size, src_seq_len]
        # tgt_padding_mask: [batch_size, tgt_seq_len]
        # memory_key_padding_mask: [batch_size, src_seq_len] (обычно совпадает с src_padding_mask)
        # tgt_mask: [tgt_seq_len, tgt_seq_len]

        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        # src_emb: [batch_size, src_seq_len, emb_size]
        # tgt_emb: [batch_size, tgt_seq_len, emb_size]

        # print(f"src_emb shape: {src_emb.shape}")
        # print(f"tgt_emb shape: {tgt_emb.shape}")
        # print(f"tgt_mask shape: {tgt_mask.shape}")
        # print(f"src_padding_mask shape: {src_padding_mask.shape}")
        # print(f"tgt_padding_mask shape: {tgt_padding_mask.shape}")
        # print(f"memory_key_padding_mask shape: {memory_key_padding_mask.shape}")


        outs = self.transformer(src_emb, tgt_emb,
                                tgt_mask=tgt_mask,
                                src_key_padding_mask=src_padding_mask,
                                tgt_key_padding_mask=tgt_padding_mask,
                                memory_key_padding_mask=memory_key_padding_mask)
        # outs: [batch_size, tgt_seq_len, emb_size]
        return self.generator(outs) # [batch_size, tgt_seq_len, tgt_vocab_size]

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        # src: [batch_size, src_seq_len]
        # src_mask: [src_seq_len, src_seq_len] (треугольная, не используется стандартно в encode)
        # src_padding_mask генерируется внутри или передается
        src_padding_mask = (src == PAD_IDX) # [batch_size, src_seq_len]
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        # print(f"Encode src_emb shape: {src_emb.shape}")
        # print(f"Encode src_padding_mask shape: {src_padding_mask.shape}")
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_padding_mask) # [batch_size, src_seq_len, emb_size]

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
        # tgt: [batch_size, tgt_seq_len]
        # memory: [batch_size, src_seq_len, emb_size] (output of encoder)
        # tgt_mask: [tgt_seq_len, tgt_seq_len]
        # tgt_padding_mask: [batch_size, tgt_seq_len]
        # memory_key_padding_mask: [batch_size, src_seq_len]

        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        # print(f"Decode tgt_emb shape: {tgt_emb.shape}")
        # print(f"Decode memory shape: {memory.shape}")
        # print(f"Decode tgt_mask shape: {tgt_mask.shape}")
        # print(f"Decode tgt_padding_mask shape: {tgt_padding_mask.shape}")
        # print(f"Decode memory_key_padding_mask shape: {memory_key_padding_mask.shape}")

        return self.transformer.decoder(tgt_emb, memory,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask) # [batch_size, tgt_seq_len, emb_size]

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask # [sz, sz]

def greedy_decode(model, src, src_padding_mask, max_len, start_symbol_idx, end_symbol_idx, device):
    model.eval()
    src = src.to(device)
    src_padding_mask = src_padding_mask.to(device)

    with torch.no_grad():
        # Энкодер вызывается один раз
        memory = model.encode(src, None) # src_mask не нужен для энкодера здесь
        # memory: [batch_size, src_seq_len, emb_size]
        memory = memory.to(device)
        # memory_key_padding_mask нужен для декодера
        memory_key_padding_mask = src_padding_mask # [batch_size, src_seq_len]

        # Инициализация выходной последовательности для декодера (начинаем с SOS)
        # Добавляем batch dimension = 1
        ys = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(device) # [1, 1]

        for i in range(max_len - 1):
            # Получаем маску для текущей длины выходной последовательности
            tgt_seq_len = ys.shape[1]
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device).to(device) # [tgt_seq_len, tgt_seq_len]
            # Паддинг маска для выхода не нужна, т.к. генерируем по одному токену
            tgt_padding_mask = torch.zeros(ys.shape, dtype=torch.bool).to(device) # [1, tgt_seq_len]

            # Декодируем
            # print(f"\nDecode Step {i}:")
            # print(f" ys shape: {ys.shape}")
            # print(f" memory shape: {memory.shape}")
            # print(f" tgt_mask shape: {tgt_mask.shape}")
            # print(f" tgt_padding_mask shape: {tgt_padding_mask.shape}")
            # print(f" memory_key_padding_mask shape: {memory_key_padding_mask.shape}")

            out = model.decode(ys, memory, tgt_mask, tgt_padding_mask=tgt_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            # out: [batch_size=1, current_tgt_seq_len, emb_size]
            # Берем последний выход по времени
            out = out[:, -1, :] # [1, emb_size]
            # Преобразуем в вероятности токенов
            prob = model.generator(out) # [1, tgt_vocab_size]
            # Выбираем самый вероятный токен (Greedy)
            _, next_word_idx = torch.max(prob, dim=1)
            next_word_idx = next_word_idx.item()

            # Добавляем предсказанный токен к последовательности
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_idx)], dim=1)

            # Если предсказан EOS, завершаем
            if next_word_idx == end_symbol_idx:
                break

    # Убираем SOS токен из результата
    return ys[0, 1:] # Возвращаем тензор токенов без batch dimension и без SOS

def reconstruct_message(corrupted_msg: str, last_valid_msg: str, model: Seq2SeqTransformer, tokenizer: CharacterTokenizer, device: torch.device, max_len: int = MAX_SEQ_LEN):
    model.eval() # Переключаем модель в режим инференса

    # 1. Подготовка входных данных
    context_tokens = tokenizer.encode(last_valid_msg, add_special_tokens=False) if last_valid_msg else []
    corrupted_tokens = tokenizer.encode(corrupted_msg, add_special_tokens=False)
    src_tokens = context_tokens + [SEP_IDX] + corrupted_tokens
    src_tokens = src_tokens[-(max_len-2):] # Обрезаем если нужно
    src_tensor = torch.tensor([src_tokens], dtype=torch.long).to(device) # Добавляем batch dimension

    # 2. Создание паддинг маски для входа
    # False там, где реальные токены, True там, где паддинг (здесь паддинга нет, т.к. 1 пример)
    src_padding_mask = torch.zeros(src_tensor.shape, dtype=torch.bool).to(device)

    # 3. Генерация выходной последовательности
    generated_token_ids = greedy_decode(
        model,
        src_tensor,
        src_padding_mask,
        max_len=max_len,
        start_symbol_idx=SOS_IDX,
        end_symbol_idx=EOS_IDX,
        device=device
    )
    # print(generated_token_ids.cpu().numpy())
    # 4. Декодирование токенов в строку
    reconstructed_msg = tokenizer.decode(generated_token_ids.cpu().numpy(), remove_special_tokens=True)

    # 5. Валидация JSON (опционально, но рекомендуется)
    try:
        json.loads(reconstructed_msg)
        print(f"Reconstruction successful and valid JSON.")
        return reconstructed_msg
    except json.JSONDecodeError as e:
        print(f"Reconstruction produced invalid JSON: {e}")
        print(f"Raw reconstructed output: {reconstructed_msg}")
        # Возможно, стоит вернуть сырой вывод, если валидация не прошла,
        # или None, чтобы показать неудачу. Зависит от требований.
        return reconstructed_msg # Возвращаем даже если невалидный JSON, чтобы посмотреть


def calculate_molar_mass(humidity):
    M_dry = 28.97  # Молярная масса сухого воздуха, г/моль
    M_water = 18.02  # Молярная масса водяного пара, г/моль
    humidity /= 100
    molar_mass = (1 - humidity) * M_dry + humidity * M_water
    # print(molar_mass)
    return molar_mass

def calculate_height(pressure, temperature, humidity):
    R = 8.3144621  # Универсальная газовая постоянная, Дж/(моль·К)
    g = 9.80665    # Ускорение свободного падения, м/с²

    # Приведение температуры к Кельвинам
    temperature_k = temperature + 273.15

    # Расчет молярной массы влажного воздуха
    molar_mass = calculate_molar_mass(humidity)

    # Расчет высоты
    height = (R * temperature_k) / (molar_mass * g) * math.log(101325 / pressure)
    # print("Высота = ", height)
    return height

class CustomQVTKRenderWindowInteractor(QVTKRenderWindowInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        pass

    def keyPressEvent(self, event):
        pass

    def keyReleaseEvent(self, event):
        pass

class TiltWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vtk_widget = CustomQVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(self)

        # Добавление QLabel для названия
        self.title_label = QLabel("Наклон", self)
        self.title_label.setStyleSheet("border: none;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setMaximumHeight(20)
        layout.addWidget(self.title_label)

        layout.addWidget(self.vtk_widget)
        self.setLayout(layout)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.stl_reader = vtk.vtkSTLReader()
        self.stl_reader.SetFileName("sat.stl")
        self.stl_reader.Update()

        self.model_mapper = vtk.vtkPolyDataMapper()
        self.model_mapper.SetInputConnection(self.stl_reader.GetOutputPort())
        self.model_actor = vtk.vtkActor()
        self.model_actor.SetMapper(self.model_mapper)
        self.renderer.AddActor(self.model_actor)
        self.renderer.SetBackground(0.176, 0.176, 0.176)
        self.renderer.ResetCamera()

        self.rotate_camera(0, -90)

        self.interactor.Initialize()
        self.interactor.Start()

        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("border: none;")
        self.info_label.setMaximumHeight(20)
        layout.addWidget(self.info_label)

    def update_orientation(self, tx, ty, tz):
        self.model_actor.SetPosition(0, 0, 0)
        transform = vtk.vtkTransform()
        transform.RotateX(ty)
        transform.RotateY(tx)
        transform.RotateZ(-tz)
        self.model_actor.SetUserTransform(transform)
        self.vtk_widget.GetRenderWindow().Render()

        # Обновление текста QLabel
        self.info_label.setText(f"x: {tx}°, y: {ty}°, z: {tz}°")

    def rotate_camera(self, azimuth, elevation):
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(azimuth)
        camera.Elevation(elevation)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()


class CompassWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vtk_widget = CustomQVTKRenderWindowInteractor(self)
        layout = QVBoxLayout(self)

        # Добавление QLabel для названия
        self.compass_label = QLabel("Компас", self)
        self.compass_label.setStyleSheet("border: none;")
        self.compass_label.setAlignment(Qt.AlignCenter)
        self.compass_label.setMaximumHeight(20)
        layout.addWidget(self.compass_label)

        layout.addWidget(self.vtk_widget)
        self.setLayout(layout)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.stl_reader = vtk.vtkSTLReader()
        self.stl_reader.SetFileName("arrow.stl")
        self.stl_reader.Update()

        self.arrow_mapper = vtk.vtkPolyDataMapper()
        self.arrow_mapper.SetInputConnection(self.stl_reader.GetOutputPort())
        self.arrow_actor = vtk.vtkActor()
        self.arrow_actor.SetMapper(self.arrow_mapper)
        self.renderer.AddActor(self.arrow_actor)

        self.renderer.SetBackground(0.176, 0.176, 0.176)
        self.renderer.ResetCamera()
        self.zoom_out_camera(0.5)

        self.rotate_camera(0, 180)

        self.interactor.Initialize()
        self.interactor.Start()

        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("border: none;")
        self.info_label.setMaximumHeight(20)
        layout.addWidget(self.info_label)

    def update_compass(self, chyx, chzx, deg):
        transform = vtk.vtkTransform()
        transform.RotateZ(chyx)
        self.arrow_actor.SetUserTransform(transform)
        self.vtk_widget.GetRenderWindow().Render()

        # Обновление текста QLabel
        self.info_label.setText(f"yx: {chyx}°, zx: {chzx}°, deg: {deg}°")

    def zoom_out_camera(self, zoom_factor=1):
        camera = self.renderer.GetActiveCamera()
        camera.Zoom(zoom_factor)  # Уменьшение масштаба камеры
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def rotate_camera(self, azimuth, elevation):
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(azimuth)
        camera.Elevation(elevation)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()



class AccelWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(facecolor='#2d2d2d')
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)

        # Добавление QLabel для названия
        self.accel_label = QLabel("Акселерометр", self)
        self.accel_label.setStyleSheet("border: none;")
        self.accel_label.setAlignment(Qt.AlignCenter)
        self.accel_label.setMaximumHeight(20)
        layout.addWidget(self.accel_label)

        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_facecolor('#2d2d2d')
        self.figure.subplots_adjust(left=-0.2, right=1.2, top=1.2, bottom=-0.2)

        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.tick_params(axis='z', colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')

        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet("border: none;")
        self.info_label.setMaximumHeight(20)
        layout.addWidget(self.info_label)

    def update_acceleration(self, ax, ay, az):
        self.ax.cla()
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.quiver(0, 0, 0, ax, ay, az, length=1, normalize=True, linewidth=3)
        self.canvas.draw()

        # Обновление текста QLabel
        self.info_label.setText(f"x: {ax} м/c/c, y: {ay} м/c/c, z: {az} м/c/c")


# Поток для чтения данных из файла
class DataReaderThread(QThread):
    data_ready = Signal(object)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for entry in data:
                data_entry = DataEntry(**entry)
                self.data_ready.emit(data_entry)
                time.sleep(0.01)


# Класс для хранения данных телеметрии
class DataEntry:
    def __init__(self, d=None, mh=None, y=None, hr=None, m=None, s=None, sc=None, hdop=None, gx=None, gy=None, gz=None,
                 ax=None, ay=None, az=None, tx=None, ty=None, tz=None, tb=None, pres=None, alt=None, salt=None, u=None,
                 i=None, p=None, cx=None, cy=None, cz=None, chyx=None, chzx=None, ad=None, dr=None, pc=None, td=None,
                 bat=None, lat=None, lng=None, mps=None, deg=None, galt=None, gs=None, c=None, t=None, h=None):
        self.d = d
        self.mh = mh
        self.y = y
        self.hr = hr
        self.m = m
        self.s = s
        self.sc = sc
        self.hdop = hdop
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.ax = ax
        self.ay = ay
        self.az = az
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.tb = tb
        self.pres = pres
        self.alt = alt
        self.salt = salt
        self.u = u
        self.i = i
        self.p = p
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.chyx = chyx
        self.chzx = chzx
        self.ad = ad
        self.dr = dr
        self.pc = pc
        self.td = td
        self.bat = bat
        self.lat = lat
        self.lng = lng
        self.mps = mps
        self.deg = deg
        self.galt = galt
        self.gs = gs
        self.c = c
        self.t = t
        self.h = h


# Поток для чтения данных из последовательного порта
class SerialReader(QThread):
    data_received = Signal(str)

    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self.running = True

    def run(self):
        while self.running:
            if self.serial_port.in_waiting > 0:
                try:
                    data = self.serial_port.readline().decode('utf-8').strip()
                    print(data)
                    self.data_received.emit(data)
                except UnicodeDecodeError:
                    print("Ошибка декодирования строки, пропуск...")
                    continue

    def stop(self):
        self.running = False
        self.serial_port.close()


class MapWidget(QWebEngineView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_map()

    def init_map(self):
        self.map_ = folium.Map(location=[55.5867, 37.25044], zoom_start=7)
        self.data = io.BytesIO()
        self.map_.save(self.data, close_file=False)
        html_content = self.data.getvalue().decode()
        html_content += """
            <div id="map" style="width: 100%%; height: 100%%;"></div>
            <script>
                var map = L.map('map').setView([%f, %f], 7);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                }).addTo(map);
                var path = [];
                window.updateMap = function(lat, lon) {
                    var newLatLng = new L.LatLng(lat, lon);
                    var exists = path.some(function(coord) {
                        return coord[0] === lat && coord[1] === lon;
                    });
                    if (!exists) {
                        path.push([lat, lon]);
                        if (path.length > 1) {
                            L.polyline(path, {color: 'red'}).addTo(map);
                        }
                        map.setView(newLatLng, map.getZoom());
                    }
                };
            </script>
        """ % (55.5867, 37.25044)
        self.setHtml(html_content)

    def update_map(self, lat, lon):
        if lat == 0 and lon == 0:
            return
        self.page().runJavaScript(f"updateMap({lat}, {lon});")


class DatabaseManager:
    def __init__(self):
        self.conn, self.cursor = self.create_database()
        self.table_name = self.create_new_table()

    def create_database(self):
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS DataEntries (
                id INTEGER PRIMARY KEY,
                d INTEGER,
                mh INTEGER,
                y INTEGER,
                hr INTEGER,
                m INTEGER,
                s INTEGER,
                sc INTEGER,
                hdop REAL,
                gx REAL,
                gy REAL,
                gz REAL,
                ax REAL,
                ay REAL,
                az REAL,
                tx REAL,
                ty REAL,
                tz REAL,
                tb REAL,
                pres INTEGER,
                alt REAL,
                salt REAL,
                u REAL,
                i REAL,
                p REAL,
                cx REAL,
                cy REAL,
                cz REAL,
                chyx REAL,
                chzx REAL,
                ad REAL,
                dr REAL,
                pc INTEGER,
                td INTEGER,
                bat INTEGER,
                lat REAL,
                lng REAL,
                mps REAL,
                deg REAL,
                galt REAL,
                gs INTEGER,
                c INTEGER,
                t REAL,
                h REAL
            )
        ''')
        conn.commit()
        return conn, cursor

    def create_new_table(self):
        table_name = datetime.now().strftime('DataEntries_%Y%m%d_%H%M%S')
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY,
                d INTEGER,
                mh INTEGER,
                y INTEGER,
                hr INTEGER,
                m INTEGER,
                s INTEGER,
                sc INTEGER,
                hdop REAL,
                gx REAL,
                gy REAL,
                gz REAL,
                ax REAL,
                ay REAL,
                az REAL,
                tx REAL,
                ty REAL,
                tz REAL,
                tb REAL,
                pres INTEGER,
                alt REAL,
                salt REAL,
                u REAL,
                i REAL,
                p REAL,
                cx REAL,
                cy REAL,
                cz REAL,
                chyx REAL,
                chzx REAL,
                ad REAL,
                dr REAL,
                pc INTEGER,
                td INTEGER,
                bat INTEGER,
                lat REAL,
                lng REAL,
                mps REAL,
                deg REAL,
                galt REAL,
                gs INTEGER,
                c INTEGER,
                t REAL,
                h REAL
            )
        ''')
        self.cursor.connection.commit()
        return table_name

    def save_to_database(self, telemetry):
        values = (
            telemetry.d, telemetry.mh, telemetry.y, telemetry.hr, telemetry.m, telemetry.s, telemetry.sc, telemetry.hdop,
            telemetry.gx, telemetry.gy, telemetry.gz, telemetry.ax, telemetry.ay, telemetry.az, telemetry.tx,
            telemetry.ty,
            telemetry.tz, telemetry.tb, telemetry.pres, telemetry.alt, telemetry.salt, telemetry.u, telemetry.i,
            telemetry.p,
            telemetry.cx, telemetry.cy, telemetry.cz, telemetry.chyx, telemetry.chzx, telemetry.ad, telemetry.dr,
            telemetry.pc,
            telemetry.td, telemetry.bat, telemetry.lat, telemetry.lng, telemetry.mps, telemetry.deg, telemetry.galt,
            telemetry.gs,
            telemetry.c, telemetry.t, telemetry.h
        )

        values = [None if v is None else v for v in values]

        self.cursor.execute(f'''
            INSERT INTO {self.table_name} (d, mh, y, hr, m, s, sc, hdop, gx, gy, gz, ax, ay, az, tx, ty, tz, tb, pres, alt, salt, u, i, p, cx, cy, cz, chyx, chzx, ad, dr, pc, td, bat, lat, lng, mps, deg, galt, gs, c, t, h)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', values)
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()


# Основное окно приложения
class TelemetryViewer(QMainWindow):
    def __init__(self):
        super(TelemetryViewer, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.serial_port = None
        self.serial_thread = None
        self.db_manager = None
        self.temp_data = {}

        self.repair_model = None
        self.repair_tokenizer = None
        self.transformer_max_len = 512
        self.last_valid_message = ""
        self.load_repair_model()

        self.init_map()
        self.init_tilt_widget()
        self.init_compass_widgets()
        self.init_accel_widget()

        self.file = False
        if not self.file:
            self.populate_com_ports()
        else:
            self.start_data_reader_thread()
        self.ui.btn_conect.clicked.connect(self.toggle_connection)

    def load_repair_model(self):
        print("Attempting to load JSON repair model...")
        if not all(os.path.exists(p) for p in [MODEL_SAVE_PATH, VOCAB_SAVE_PATH, TOKENIZER_SAVE_PATH]):
            print(f"Warning: Model/Vocab/Config files not found ({MODEL_SAVE_PATH}, {VOCAB_SAVE_PATH}, {TOKENIZER_SAVE_PATH}). Repair feature disabled.")
            return

        try:
            self.repair_tokenizer = CharacterTokenizer.load(VOCAB_SAVE_PATH, TOKENIZER_SAVE_PATH)

            # Load checkpoint and model parameters
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            model_params = checkpoint['params']

            # Update MAX_SEQ_LEN from the loaded model if available
            self.transformer_max_len = model_params.get('max_seq_len', 512)
            print(f"Model max_seq_len set to: {self.transformer_max_len}")

            # Ensure all required parameters are present
            required_keys = ['num_encoder_layers', 'num_decoder_layers', 'emb_size', 'nhead',
                             'src_vocab_size', 'tgt_vocab_size', 'dim_feedforward', 'dropout']
            if not all(key in model_params for key in required_keys):
                raise ValueError(f"Checkpoint missing required parameters. Found: {model_params.keys()}")

            # Adjust vocab size based on loaded tokenizer, not just checkpoint param
            loaded_src_vocab_size = len(self.repair_tokenizer.char2idx)
            if loaded_src_vocab_size != model_params['src_vocab_size']:
                print(f"Warning: Tokenizer vocab size ({loaded_src_vocab_size}) differs from model checkpoint ({model_params['src_vocab_size']}). Using tokenizer size.")

            # Instantiate the model
            self.repair_model = Seq2SeqTransformer(
                num_encoder_layers=model_params['num_encoder_layers'],
                num_decoder_layers=model_params['num_decoder_layers'],
                emb_size=model_params['emb_size'],
                nhead=model_params['nhead'],
                # Use the actual size from the loaded tokenizer
                src_vocab_size=loaded_src_vocab_size,
                tgt_vocab_size=loaded_src_vocab_size,
                dim_feedforward=model_params['dim_feedforward'],
                dropout=model_params['dropout'],
                max_seq_len=self.transformer_max_len # Pass max_len here
            )

            # Load the trained weights
            self.repair_model.load_state_dict(checkpoint['model_state_dict'])
            self.repair_model.to(DEVICE)
            self.repair_model.eval() # Set to evaluation mode

            print(f"Transformer repair model loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}.")

        except FileNotFoundError:
            print(f"Warning: Failed to find model/tokenizer files. Repair feature disabled.")
            self.repair_model = None
            self.repair_tokenizer = None
        except Exception as e:
            print(f"Error loading repair model: {e}. Repair feature disabled.")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            self.repair_model = None
            self.repair_tokenizer = None

    def init_tilt_widget(self):
        self.tilt_widget = TiltWidget(self.ui.tilt_widget)
        layout = QVBoxLayout(self.ui.tilt_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tilt_widget)
        self.ui.tilt_widget.setLayout(layout)

    def init_compass_widgets(self):
        self.compass_widget_compass = CompassWidget(self.ui.compass_widget)
        layout = QVBoxLayout(self.ui.compass_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.compass_widget_compass)
        self.ui.compass_widget.setLayout(layout)
        self.update_compass_widget_size()

    def init_accel_widget(self):
        self.accel_widget = AccelWidget(self.ui.accel_widget)
        layout = QVBoxLayout(self.ui.accel_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.accel_widget)
        self.ui.accel_widget.setLayout(layout)
        self.update_accel_widget_size()

    # Запуск потока для чтения данных из файла
    def start_data_reader_thread(self):
        self.data_reader_thread = DataReaderThread('exp4.json')
        self.data_reader_thread.data_ready.connect(self.process_data_entry)
        self.data_reader_thread.start()

    # Обработка полученных данных
    def process_data_entry(self, data_entry):
        self.update_ui(data_entry)

    # Заполнение списка доступных COM портов
    def populate_com_ports(self):
        self.ui.cb_port.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.ui.cb_port.addItem(port.device)

    # Переключение состояния подключения
    def toggle_connection(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_thread.stop()
            self.serial_thread.wait()
            self.serial_port = None
            self.update_connection_status(False)
        else:
            selected_port = self.ui.cb_port.currentText()
            if not selected_port:
                QMessageBox.warning(self, "Ошибка", "Выберите COM порт.")
                return

            try:
                self.serial_port = serial.Serial(selected_port, baudrate=9600, timeout=1)
                self.serial_thread = SerialReader(self.serial_port)
                self.serial_thread.data_received.connect(self.process_data)
                self.serial_thread.start()
                self.db_manager = DatabaseManager()
                self.update_connection_status(True)
            except serial.SerialException as e:
                QMessageBox.critical(self, "Ошибка подключения", f"Не удалось подключиться к порту: {e}")

    # Обновление статуса подключения
    def update_connection_status(self, connected):
        if connected:
            self.ui.val_status_conect.setStyleSheet(u"border-radius: 7;\n"
                                                    "background-color: rgb(0, 170, 0);")
            self.ui.btn_conect.setText("Отключиться")
        else:
            if self.db_manager:
                self.db_manager.close()
                self.db_manager = None
            self.ui.val_status_conect.setStyleSheet(u"border-radius: 7;\n"
                                                    "background-color: rgb(170, 0, 0);")
            self.ui.btn_conect.setText("Подключиться")

    # Обработка данных, полученных из последовательного порта
    def process_data(self, data):
        original_data = data.strip() # Ensure no leading/trailing whitespace
        repaired_data = None
        is_repaired = False

        if self.repair_model and self.repair_tokenizer and original_data:
            print(f"Received raw: {original_data}")
            print(f"Context: {self.last_valid_message[:50]}...") # Print context being used
            repaired_data = reconstruct_message(
                corrupted_msg=original_data,
                last_valid_msg=self.last_valid_message,
                model=self.repair_model,
                tokenizer=self.repair_tokenizer,
                device=DEVICE,
                max_len=self.transformer_max_len
            )
            if repaired_data:
                print(f"Repair attempt result: {repaired_data}")
                is_repaired = True
            else:
                print("Repair attempt failed or produced invalid output.")
        else:
            # If model not loaded or data is empty, use original
            repaired_data = original_data
            print(f"Skipping repair for: {original_data}")

        final_data_to_parse = None
        parsed_successfully = False
        parsed_dict = None

        if is_repaired and repaired_data:
            try:
                parsed_dict = json.loads(repaired_data)
                final_data_to_parse = repaired_data # Keep track of what worked
                parsed_successfully = True
                print("Successfully parsed repaired data.")
            except json.JSONDecodeError as e1:
                print(f"Failed to parse repaired data ('{repaired_data}'): {e1}. Trying original.")
                # Fall through to try original

        if not parsed_successfully:
            try:
                parsed_dict = json.loads(original_data)
                final_data_to_parse = original_data # Keep track of what worked
                parsed_successfully = True
                print("Successfully parsed original data.")
            except json.JSONDecodeError as e2:
                print(f"Failed to parse original data either ('{original_data}'): {e2}. Skipping message.")
                self.temp_data.clear() # Clear temp buffer as message is unusable
                return # Stop processing this message

        if parsed_successfully and parsed_dict is not None:
            # Update context ONLY if the data we successfully parsed was the REPAIRED one,
            # or if the original data parsed correctly and no repair was attempted/needed.
            # This prevents using potentially corrupted original data as future context if repair failed.
            if final_data_to_parse == repaired_data:
                self.last_valid_message = final_data_to_parse
                print("Updated context with repaired message.")
            elif final_data_to_parse == original_data and not is_repaired:
                self.last_valid_message = final_data_to_parse
                print("Updated context with original message (repair not attempted/needed).")
            elif final_data_to_parse == original_data and is_repaired:
                print("Did not update context (used original data after repair failed parsing).")


            self.temp_data.update(parsed_dict) # Update temporary storage with parsed data

            # Check for required keys (using the parsed_dict)
            # Define required keys based on what update_ui actually uses
            required_keys_short = {"pc", "pres", "t", "h", "lat", "lng", "galt", "c"} # Keys from the shorter message type
            required_keys_long = {"dr", "ad", "td", "bat", "dt", "dh", "mps", "deg", "chyx", "chzx",
                                  "tx", "ty", "tz", "tb", "alt", "salt", "cx", "cy", "cz", "gx",
                                  "gy", "gz", "ax", "ay", "az", "gs", "hdop", "sc", "d", "mh",
                                  "y", "hr", "m", "s", "u", "i", "p", "c"} # Keys from the longer message type

            current_keys = self.temp_data.keys()

            # Check if we have enough keys for *either* type of message processing
            # Or better: check if *all* keys used by update_ui are present before calling it.
            # Let's assume update_ui handles missing keys gracefully for now.
            # A simple check could be based on 'pc' presence:
            is_complete = False
            if 'pc' in current_keys: # Likely short message type
                if required_keys_short.issubset(current_keys):
                    is_complete = True
            else: # Likely long message type
                if required_keys_long.issubset(current_keys):
                    is_complete = True

            # If we consider the message complete (or decide to process partial data)
            # Convert to DataEntry, update UI, save to DB
            if is_complete: # Or remove this check if you process partial messages
                print("Processing complete message packet.")
                try:
                    # Ensure all expected fields exist, filling with None if necessary
                    # Create a complete dict with all possible keys set to None initially
                    all_possible_keys = required_keys_short.union(required_keys_long)
                    complete_data = {key: None for key in all_possible_keys}
                    complete_data.update(self.temp_data) # Overwrite Nones with actual values

                    telemetry = DataEntry(**complete_data)
                    self.update_ui(telemetry)
                    if self.db_manager: # Only save if connected
                        self.db_manager.save_to_database(telemetry)
                    self.temp_data.clear() # Clear buffer after successful processing
                    print("Cleared temp data buffer.")
                except Exception as e:
                    print(f"Error processing complete data entry: {e}")
                    self.temp_data.clear() # Clear buffer on error too
            else:
                print("Incomplete message packet in buffer, waiting for more data.")
        else:
            # This case should ideally not be reached if the parsing logic above is correct
            print("Message skipped due to parsing failure.")
            self.temp_data.clear()


    def update_gps_status(self, gs):
        status_colors = {
            48: "rgb(170, 0, 0)",  # Invalid - красный
            49: "rgb(0, 85, 0)",  # GPS - темно-зеленый
            50: "rgb(0, 128, 0)",  # DGPS - средне-зеленый
            51: "rgb(0, 170, 0)",  # PPS - светло-зеленый
            52: "rgb(0, 255, 0)",  # RTK - ярко-зеленый
            53: "rgb(170, 170, 0)",  # FloatRTK - темно-желтый
            54: "rgb(255, 255, 0)",  # Estimated - желтый
            55: "rgb(255, 255, 0)",  # Manual - желтый
            56: "rgb(255, 255, 0)",  # Simulated - желтый
        }
        color = status_colors.get(gs)  # По умолчанию красный
        self.ui.val_status_gps.setStyleSheet(f"border-radius: 7;\nbackground-color: {color};")

    # Обновление пользовательского интерфейса
    def update_ui(self, telemetry):
        if telemetry.alt is not None:
            self.ui.val_alt.setText(f"Высота: {telemetry.alt} м")
        if telemetry.t is not None:
            self.ui.val_sea_temp.setText(f"Температура: {telemetry.t} °C")
        if telemetry.h is not None:
            self.ui.val_sea_hum.setText(f"Влажность: {telemetry.h} %")
        if telemetry.salt is not None:
            self.ui.val_sea_alt.setText(f"Абсолютная высота: {telemetry.salt} м")
        if telemetry.pres is not None and telemetry.t is not None and telemetry.h is not None:
            height = calculate_height(telemetry.pres, telemetry.t, telemetry.h)
            self.ui.val_calc_alt.setText(f"Расчетная высота: {height:.3f} м")
        if telemetry.dr is not None:
            self.ui.val_dose_rate.setText(f"Доза: {telemetry.dr} мкЗв/ч")
        if telemetry.d is not None and telemetry.mh is not None and telemetry.y is not None:
            self.ui.val_date.setText(f"{telemetry.d}.{telemetry.mh}.{telemetry.y}")
        if telemetry.hr is not None and telemetry.m is not None and telemetry.s is not None:
            self.ui.val_time.setText(f"{telemetry.hr}:{telemetry.m}:{telemetry.s}")
        if telemetry.lat is not None:
            self.ui.val_lat.setText(f"Широта: {telemetry.lat}°")
        if telemetry.lng is not None:
            self.ui.val_lon.setText(f"Долгота: {telemetry.lng}°")
        if telemetry.bat is not None:
            self.ui.val_dos_bat.setText(f"    Заряд дозиметра: {telemetry.bat}%")
        if telemetry.u is not None:
            self.ui.val_volt.setText(f"Напряжение: {telemetry.u} В")
        if telemetry.i is not None:
            self.ui.val_cur.setText(f"Ток: {telemetry.i / 1000} А")
        if telemetry.galt is not None:
            self.ui.val_gps_alt.setText(f"Высота по GPS: {telemetry.galt} м")
        if telemetry.mps is not None:
            self.ui.val_speed.setText(f"Скорость: {telemetry.mps} м/с")
        if telemetry.c is not None:
            self.ui.val_msg_count.setText(f"    Сообщейний: {telemetry.c}")
        if telemetry.sc is not None:
            self.ui.val_satelits_count.setText(f"    Cпутников GPS: {telemetry.sc}")
        if telemetry.lat is not None and telemetry.lng is not None:
            self.map_widget.update_map(telemetry.lat, telemetry.lng)
        if telemetry.gs is not None:
            self.update_gps_status(telemetry.gs)
        if telemetry.chyx is not None and telemetry.chzx is not None and telemetry.deg is not None:
            self.compass_widget_compass.update_compass(telemetry.chyx, telemetry.chzx, telemetry.deg)
        if telemetry.tx is not None and telemetry.ty is not None and telemetry.tz is not None:
            self.tilt_widget.update_orientation(telemetry.tx, telemetry.ty, telemetry.tz)
        if telemetry.ax is not None and telemetry.ay is not None and telemetry.az is not None:
            self.accel_widget.update_acceleration(telemetry.ax, telemetry.ay, -telemetry.az)

    # Инициализация карты
    def init_map(self):
        self.map_widget = MapWidget(self.ui.map_widget)
        layout = QVBoxLayout(self.ui.map_widget)
        self.ui.map_widget.setLayout(layout)
        layout.addWidget(self.map_widget)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_compass_widget_size()
        self.update_accel_widget_size()

    def update_compass_widget_size(self):
        map_width = self.ui.map_widget.width()
        self.ui.compass_widget.setFixedWidth(map_width // 3)

    def update_accel_widget_size(self):
        map_width = self.ui.map_widget.width()
        self.ui.accel_widget.setFixedWidth(map_width // 3)

    def closeEvent(self, event):
        print("Closing application...")
        if self.serial_thread:
            print("Stopping serial thread...")
            self.serial_thread.stop()
            self.serial_thread.wait()
            print("Serial thread stopped.")
        if hasattr(self, 'data_reader_thread') and self.data_reader_thread and self.data_reader_thread.isRunning():
            print("Terminating data reader thread...")
            self.data_reader_thread.terminate() # Use terminate for file reading thread if needed
            self.data_reader_thread.wait()
            print("Data reader thread terminated.")
        if self.db_manager:
            print("Closing database connection...")
            self.db_manager.close()
            print("Database connection closed.")

        # VTK Cleanup (Add checks if widgets exist)
        print("Cleaning up VTK resources...")
        if hasattr(self, 'tilt_widget') and self.tilt_widget.interactor:
            try:
                render_window = self.tilt_widget.interactor.GetRenderWindow()
                if render_window: render_window.Finalize() # Use Finalize
                # self.tilt_widget.interactor.TerminateApp() # Avoid TerminateApp if possible
                self.tilt_widget.vtk_widget.close() # Close the widget
            except Exception as e: print(f"Error cleaning tilt widget: {e}")

        if hasattr(self, 'compass_widget_compass') and self.compass_widget_compass.interactor:
            try:
                render_window = self.compass_widget_compass.interactor.GetRenderWindow()
                if render_window: render_window.Finalize()
                # self.compass_widget_compass.interactor.TerminateApp()
                self.compass_widget_compass.vtk_widget.close()
            except Exception as e: print(f"Error cleaning compass widget: {e}")

        print("Cleanup complete.")
        event.accept()


# Запуск приложения
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TelemetryViewer()
    window.showMaximized()
    sys.exit(app.exec())
