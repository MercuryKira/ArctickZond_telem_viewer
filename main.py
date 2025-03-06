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
        try:
            telemetry_dict = json.loads(data)
            self.temp_data.update(telemetry_dict)  # Обновление временного хранилища

            # Проверка наличия всех необходимых данных
            required_keys = {"dr", "ad", "td", "bat", "mps", "deg", "chyx", "chzx", "tx", "ty", "tz", "tb", "alt", "salt", "cx", "cy", "cz", "gx", "gy", "gz", "ax", "ay", "az", "gs", "sc", "d", "mh", "y", "hr", "m", "s", "u", "i", "p", "c", "pc", "pres", "t", "h", "lat", "lng", "galt"}
            if required_keys.issubset(self.temp_data.keys()):
                telemetry = DataEntry(**self.temp_data)
                self.update_ui(telemetry)
                self.db_manager.save_to_database(telemetry)
                self.temp_data.clear()  # Очистка временного хранилища после записи в базу данных
        except json.JSONDecodeError:
            print("Ошибка декодирования JSON")

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
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread.wait()
        if hasattr(self, 'data_reader_thread') and self.data_reader_thread:
            self.data_reader_thread.terminate()
        if self.db_manager:
            self.db_manager.close()
        if self.tilt_widget.interactor:
            render_window = self.tilt_widget.interactor.GetRenderWindow()
            render_window.ReleaseGraphicsResources(None)
            self.tilt_widget.interactor.TerminateApp()
        if hasattr(self, 'compass_widget_compass') and self.compass_widget_compass.interactor:
            render_window = self.compass_widget_compass.interactor.GetRenderWindow()
            render_window.ReleaseGraphicsResources(None)
            self.compass_widget_compass.interactor.TerminateApp()
        event.accept()


# Запуск приложения
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TelemetryViewer()
    window.showMaximized()
    sys.exit(app.exec())
