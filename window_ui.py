# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'dashboard.ui'
##
## Created by: Qt User Interface Compiler version 6.8.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
    QHBoxLayout, QLabel, QLayout, QMainWindow,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(950, 919)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.conection_menu = QFrame(self.centralwidget)
        self.conection_menu.setObjectName(u"conection_menu")
        self.conection_menu.setStyleSheet(u"border-bottom: 5px groove #3c3c3c;")
        self.port_conection = QHBoxLayout(self.conection_menu)
        self.port_conection.setSpacing(0)
        self.port_conection.setObjectName(u"port_conection")
        self.port_conection.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.label = QLabel(self.conection_menu)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(16777215, 30))
        self.label.setStyleSheet(u"font-size: 14pt;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.port_conection.addWidget(self.label)

        self.cb_port = QComboBox(self.conection_menu)
        self.cb_port.setObjectName(u"cb_port")
        self.cb_port.setMaximumSize(QSize(200, 30))

        self.port_conection.addWidget(self.cb_port)

        self.btn_conect = QPushButton(self.conection_menu)
        self.btn_conect.setObjectName(u"btn_conect")
        self.btn_conect.setMaximumSize(QSize(150, 30))
        self.btn_conect.setStyleSheet(u"font-size: 14pt;")

        self.port_conection.addWidget(self.btn_conect)


        self.verticalLayout.addWidget(self.conection_menu)

        self.main_body = QHBoxLayout()
        self.main_body.setObjectName(u"main_body")
        self.text_status_2 = QFrame(self.centralwidget)
        self.text_status_2.setObjectName(u"text_status_2")
        self.text_status_2.setMinimumSize(QSize(300, 0))
        self.text_status_2.setMaximumSize(QSize(300, 16777215))
        self.text_status = QVBoxLayout(self.text_status_2)
        self.text_status.setObjectName(u"text_status")
        self.horizontalFrame_2 = QFrame(self.text_status_2)
        self.horizontalFrame_2.setObjectName(u"horizontalFrame_2")
        self.horizontalFrame_2.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_1 = QHBoxLayout(self.horizontalFrame_2)
        self.horizontalLayout_1.setSpacing(3)
        self.horizontalLayout_1.setObjectName(u"horizontalLayout_1")
        self.horizontalLayout_1.setContentsMargins(3, 1, 3, 5)
        self.val_dose_rate = QLabel(self.horizontalFrame_2)
        self.val_dose_rate.setObjectName(u"val_dose_rate")
        self.val_dose_rate.setMinimumSize(QSize(90, 0))
        self.val_dose_rate.setMaximumSize(QSize(16777215, 16777215))
        self.val_dose_rate.setStyleSheet(u"border: none")

        self.horizontalLayout_1.addWidget(self.val_dose_rate)


        self.text_status.addWidget(self.horizontalFrame_2)

        self.horizontalFrame = QFrame(self.text_status_2)
        self.horizontalFrame.setObjectName(u"horizontalFrame")
        self.horizontalFrame.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_0 = QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout_0.setSpacing(3)
        self.horizontalLayout_0.setObjectName(u"horizontalLayout_0")
        self.horizontalLayout_0.setContentsMargins(3, 1, 3, 5)
        self.val_alt = QLabel(self.horizontalFrame)
        self.val_alt.setObjectName(u"val_alt")
        self.val_alt.setMinimumSize(QSize(90, 0))
        self.val_alt.setMaximumSize(QSize(16777215, 16777215))
        self.val_alt.setStyleSheet(u"border: none")

        self.horizontalLayout_0.addWidget(self.val_alt)


        self.text_status.addWidget(self.horizontalFrame)

        self.horizontalFrame_7 = QFrame(self.text_status_2)
        self.horizontalFrame_7.setObjectName(u"horizontalFrame_7")
        self.horizontalFrame_7.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_7 = QHBoxLayout(self.horizontalFrame_7)
        self.horizontalLayout_7.setSpacing(3)
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(3, 1, 3, 5)
        self.val_gps_alt = QLabel(self.horizontalFrame_7)
        self.val_gps_alt.setObjectName(u"val_gps_alt")
        self.val_gps_alt.setMinimumSize(QSize(90, 0))
        self.val_gps_alt.setMaximumSize(QSize(16777215, 16777215))
        self.val_gps_alt.setStyleSheet(u"border: none")

        self.horizontalLayout_7.addWidget(self.val_gps_alt)


        self.text_status.addWidget(self.horizontalFrame_7)

        self.horizontalFrame_9 = QFrame(self.text_status_2)
        self.horizontalFrame_9.setObjectName(u"horizontalFrame_9")
        self.horizontalFrame_9.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_9 = QHBoxLayout(self.horizontalFrame_9)
        self.horizontalLayout_9.setSpacing(3)
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(3, 1, 3, 5)
        self.val_sea_alt = QLabel(self.horizontalFrame_9)
        self.val_sea_alt.setObjectName(u"val_sea_alt")
        self.val_sea_alt.setMinimumSize(QSize(90, 0))
        self.val_sea_alt.setMaximumSize(QSize(16777215, 16777215))
        self.val_sea_alt.setStyleSheet(u"border: none")

        self.horizontalLayout_9.addWidget(self.val_sea_alt)


        self.text_status.addWidget(self.horizontalFrame_9)

        self.horizontalFrame_17 = QFrame(self.text_status_2)
        self.horizontalFrame_17.setObjectName(u"horizontalFrame_17")
        self.horizontalFrame_17.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_14 = QHBoxLayout(self.horizontalFrame_17)
        self.horizontalLayout_14.setSpacing(3)
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setContentsMargins(3, 1, 3, 5)
        self.val_calc_alt = QLabel(self.horizontalFrame_17)
        self.val_calc_alt.setObjectName(u"val_calc_alt")
        self.val_calc_alt.setMinimumSize(QSize(90, 0))
        self.val_calc_alt.setMaximumSize(QSize(16777215, 16777215))
        self.val_calc_alt.setStyleSheet(u"border: none")

        self.horizontalLayout_14.addWidget(self.val_calc_alt)


        self.text_status.addWidget(self.horizontalFrame_17)

        self.horizontalFrame_15 = QFrame(self.text_status_2)
        self.horizontalFrame_15.setObjectName(u"horizontalFrame_15")
        self.horizontalFrame_15.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_12 = QHBoxLayout(self.horizontalFrame_15)
        self.horizontalLayout_12.setSpacing(3)
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setContentsMargins(3, 1, 3, 5)
        self.val_sea_temp = QLabel(self.horizontalFrame_15)
        self.val_sea_temp.setObjectName(u"val_sea_temp")
        self.val_sea_temp.setMinimumSize(QSize(90, 0))
        self.val_sea_temp.setMaximumSize(QSize(16777215, 16777215))
        self.val_sea_temp.setStyleSheet(u"border: none")

        self.horizontalLayout_12.addWidget(self.val_sea_temp)


        self.text_status.addWidget(self.horizontalFrame_15)

        self.horizontalFrame_16 = QFrame(self.text_status_2)
        self.horizontalFrame_16.setObjectName(u"horizontalFrame_16")
        self.horizontalFrame_16.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_13 = QHBoxLayout(self.horizontalFrame_16)
        self.horizontalLayout_13.setSpacing(3)
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setContentsMargins(3, 1, 3, 5)
        self.val_sea_hum = QLabel(self.horizontalFrame_16)
        self.val_sea_hum.setObjectName(u"val_sea_hum")
        self.val_sea_hum.setMinimumSize(QSize(90, 0))
        self.val_sea_hum.setMaximumSize(QSize(16777215, 16777215))
        self.val_sea_hum.setStyleSheet(u"border: none")

        self.horizontalLayout_13.addWidget(self.val_sea_hum)


        self.text_status.addWidget(self.horizontalFrame_16)

        self.horizontalFrame_8 = QFrame(self.text_status_2)
        self.horizontalFrame_8.setObjectName(u"horizontalFrame_8")
        self.horizontalFrame_8.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_8 = QHBoxLayout(self.horizontalFrame_8)
        self.horizontalLayout_8.setSpacing(3)
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(3, 1, 3, 5)
        self.val_speed = QLabel(self.horizontalFrame_8)
        self.val_speed.setObjectName(u"val_speed")
        self.val_speed.setMinimumSize(QSize(90, 0))
        self.val_speed.setMaximumSize(QSize(16777215, 16777215))
        self.val_speed.setStyleSheet(u"border: none")

        self.horizontalLayout_8.addWidget(self.val_speed)


        self.text_status.addWidget(self.horizontalFrame_8)

        self.horizontalFrame_4 = QFrame(self.text_status_2)
        self.horizontalFrame_4.setObjectName(u"horizontalFrame_4")
        self.horizontalFrame_4.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_3 = QHBoxLayout(self.horizontalFrame_4)
        self.horizontalLayout_3.setSpacing(3)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(3, 1, 3, 5)
        self.val_volt = QLabel(self.horizontalFrame_4)
        self.val_volt.setObjectName(u"val_volt")
        self.val_volt.setMinimumSize(QSize(90, 0))
        self.val_volt.setMaximumSize(QSize(16777215, 16777215))
        self.val_volt.setStyleSheet(u"border: none")

        self.horizontalLayout_3.addWidget(self.val_volt)


        self.text_status.addWidget(self.horizontalFrame_4)

        self.horizontalFrame_5 = QFrame(self.text_status_2)
        self.horizontalFrame_5.setObjectName(u"horizontalFrame_5")
        self.horizontalFrame_5.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalFrame_5)
        self.horizontalLayout_4.setSpacing(3)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(3, 1, 3, 5)
        self.val_cur = QLabel(self.horizontalFrame_5)
        self.val_cur.setObjectName(u"val_cur")
        self.val_cur.setMinimumSize(QSize(90, 0))
        self.val_cur.setMaximumSize(QSize(16777215, 16777215))
        self.val_cur.setStyleSheet(u"border: none")

        self.horizontalLayout_4.addWidget(self.val_cur)


        self.text_status.addWidget(self.horizontalFrame_5)

        self.horizontalFrame_6 = QFrame(self.text_status_2)
        self.horizontalFrame_6.setObjectName(u"horizontalFrame_6")
        self.horizontalFrame_6.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_5 = QHBoxLayout(self.horizontalFrame_6)
        self.horizontalLayout_5.setSpacing(3)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(3, 1, 3, 5)
        self.val_lat = QLabel(self.horizontalFrame_6)
        self.val_lat.setObjectName(u"val_lat")
        self.val_lat.setMinimumSize(QSize(90, 0))
        self.val_lat.setMaximumSize(QSize(16777215, 16777215))
        self.val_lat.setStyleSheet(u"border: none")

        self.horizontalLayout_5.addWidget(self.val_lat)


        self.text_status.addWidget(self.horizontalFrame_6)

        self.horizontalFrame_14 = QFrame(self.text_status_2)
        self.horizontalFrame_14.setObjectName(u"horizontalFrame_14")
        self.horizontalFrame_14.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_6 = QHBoxLayout(self.horizontalFrame_14)
        self.horizontalLayout_6.setSpacing(3)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(3, 1, 3, 5)
        self.val_lon = QLabel(self.horizontalFrame_14)
        self.val_lon.setObjectName(u"val_lon")
        self.val_lon.setMinimumSize(QSize(90, 0))
        self.val_lon.setMaximumSize(QSize(16777215, 16777215))
        self.val_lon.setStyleSheet(u"border: none")

        self.horizontalLayout_6.addWidget(self.val_lon)


        self.text_status.addWidget(self.horizontalFrame_14)

        self.horizontalFrame_12 = QFrame(self.text_status_2)
        self.horizontalFrame_12.setObjectName(u"horizontalFrame_12")
        self.horizontalFrame_12.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_10 = QHBoxLayout(self.horizontalFrame_12)
        self.horizontalLayout_10.setSpacing(3)
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setContentsMargins(3, 1, 3, 5)
        self.val_date = QLabel(self.horizontalFrame_12)
        self.val_date.setObjectName(u"val_date")
        self.val_date.setMinimumSize(QSize(90, 0))
        self.val_date.setMaximumSize(QSize(16777215, 16777215))
        self.val_date.setStyleSheet(u"border: none")

        self.horizontalLayout_10.addWidget(self.val_date)


        self.text_status.addWidget(self.horizontalFrame_12)

        self.horizontalFrame_13 = QFrame(self.text_status_2)
        self.horizontalFrame_13.setObjectName(u"horizontalFrame_13")
        self.horizontalFrame_13.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout_11 = QHBoxLayout(self.horizontalFrame_13)
        self.horizontalLayout_11.setSpacing(3)
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setContentsMargins(3, 1, 3, 5)
        self.val_time = QLabel(self.horizontalFrame_13)
        self.val_time.setObjectName(u"val_time")
        self.val_time.setMinimumSize(QSize(90, 0))
        self.val_time.setMaximumSize(QSize(16777215, 16777215))
        self.val_time.setStyleSheet(u"border: none")

        self.horizontalLayout_11.addWidget(self.val_time)


        self.text_status.addWidget(self.horizontalFrame_13)

        self.horizontalFrame_11 = QFrame(self.text_status_2)
        self.horizontalFrame_11.setObjectName(u"horizontalFrame_11")
        self.horizontalFrame_11.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")
        self.horizontalLayout = QHBoxLayout(self.horizontalFrame_11)
        self.horizontalLayout.setSpacing(3)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(3, 1, 3, 5)
        self.label_3 = QLabel(self.horizontalFrame_11)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setStyleSheet(u"border: none")
        self.label_3.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)

        self.horizontalLayout.addWidget(self.label_3)

        self.nn_switcher = QCheckBox(self.horizontalFrame_11)
        self.nn_switcher.setObjectName(u"nn_switcher")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nn_switcher.sizePolicy().hasHeightForWidth())
        self.nn_switcher.setSizePolicy(sizePolicy)
        self.nn_switcher.setMinimumSize(QSize(30, 30))
        self.nn_switcher.setMaximumSize(QSize(30, 30))
        self.nn_switcher.setStyleSheet(u"border: none")

        self.horizontalLayout.addWidget(self.nn_switcher)


        self.text_status.addWidget(self.horizontalFrame_11)

        self.horizontalFrame_10 = QFrame(self.text_status_2)
        self.horizontalFrame_10.setObjectName(u"horizontalFrame_10")
        self.horizontalLayout_99 = QHBoxLayout(self.horizontalFrame_10)
        self.horizontalLayout_99.setObjectName(u"horizontalLayout_99")

        self.text_status.addWidget(self.horizontalFrame_10)

        self.text_status.setStretch(15, 1)

        self.main_body.addWidget(self.text_status_2)

        self.widgets = QVBoxLayout()
        self.widgets.setObjectName(u"widgets")
        self.acel_gyro_compass_2 = QFrame(self.centralwidget)
        self.acel_gyro_compass_2.setObjectName(u"acel_gyro_compass_2")
        self.acel_gyro_compass = QHBoxLayout(self.acel_gyro_compass_2)
        self.acel_gyro_compass.setObjectName(u"acel_gyro_compass")
        self.tilt_widget = QWidget(self.acel_gyro_compass_2)
        self.tilt_widget.setObjectName(u"tilt_widget")
        sizePolicy.setHeightForWidth(self.tilt_widget.sizePolicy().hasHeightForWidth())
        self.tilt_widget.setSizePolicy(sizePolicy)
        self.tilt_widget.setMinimumSize(QSize(200, 200))
        self.tilt_widget.setSizeIncrement(QSize(0, 0))
        self.tilt_widget.setBaseSize(QSize(0, 0))
        self.tilt_widget.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")

        self.acel_gyro_compass.addWidget(self.tilt_widget)

        self.accel_widget = QWidget(self.acel_gyro_compass_2)
        self.accel_widget.setObjectName(u"accel_widget")
        self.accel_widget.setMinimumSize(QSize(200, 200))
        self.accel_widget.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")

        self.acel_gyro_compass.addWidget(self.accel_widget)

        self.compass_widget = QWidget(self.acel_gyro_compass_2)
        self.compass_widget.setObjectName(u"compass_widget")
        self.compass_widget.setMinimumSize(QSize(200, 200))
        self.compass_widget.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")

        self.acel_gyro_compass.addWidget(self.compass_widget)


        self.widgets.addWidget(self.acel_gyro_compass_2)

        self.map_widget = QWidget(self.centralwidget)
        self.map_widget.setObjectName(u"map_widget")
        self.map_widget.setMinimumSize(QSize(600, 600))
        self.map_widget.setStyleSheet(u"background-color: #2d2d2d;\n"
"border: 1px solid #3c3c3c;\n"
"border-radius: 7px;\n"
"font-size: 14pt;")

        self.widgets.addWidget(self.map_widget)

        self.widgets.setStretch(1, 70)

        self.main_body.addLayout(self.widgets)

        self.main_body.setStretch(1, 1)

        self.verticalLayout.addLayout(self.main_body)

        self.botom_status_bar = QFrame(self.centralwidget)
        self.botom_status_bar.setObjectName(u"botom_status_bar")
        self.botom_status_bar.setStyleSheet(u"font-size: 12pt;")
        self.status_bar = QHBoxLayout(self.botom_status_bar)
        self.status_bar.setSpacing(0)
        self.status_bar.setObjectName(u"status_bar")
        self.status_bar.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)
        self.status_bar.setContentsMargins(0, 0, 0, 0)
        self.label_5 = QLabel(self.botom_status_bar)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMinimumSize(QSize(130, 0))
        self.label_5.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.status_bar.addWidget(self.label_5)

        self.val_status_conect = QLabel(self.botom_status_bar)
        self.val_status_conect.setObjectName(u"val_status_conect")
        self.val_status_conect.setMinimumSize(QSize(14, 14))
        self.val_status_conect.setMaximumSize(QSize(14, 14))
        self.val_status_conect.setStyleSheet(u"border-radius: 7;\n"
"background-color: #2d2d2d;")

        self.status_bar.addWidget(self.val_status_conect)

        self.val_msg_count = QLabel(self.botom_status_bar)
        self.val_msg_count.setObjectName(u"val_msg_count")

        self.status_bar.addWidget(self.val_msg_count)

        self.label_12 = QLabel(self.botom_status_bar)
        self.label_12.setObjectName(u"label_12")
        self.label_12.setMinimumSize(QSize(0, 0))
        self.label_12.setMaximumSize(QSize(16777215, 16777215))
        self.label_12.setStyleSheet(u"border: none")
        self.label_12.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.status_bar.addWidget(self.label_12)

        self.val_status_gps = QLabel(self.botom_status_bar)
        self.val_status_gps.setObjectName(u"val_status_gps")
        self.val_status_gps.setMinimumSize(QSize(14, 14))
        self.val_status_gps.setMaximumSize(QSize(14, 14))
        self.val_status_gps.setStyleSheet(u"border-radius: 7;\n"
"background-color: #2d2d2d;")
        self.val_status_gps.setMargin(0)

        self.status_bar.addWidget(self.val_status_gps)

        self.val_satelits_count = QLabel(self.botom_status_bar)
        self.val_satelits_count.setObjectName(u"val_satelits_count")
        self.val_satelits_count.setMinimumSize(QSize(0, 0))
        self.val_satelits_count.setMaximumSize(QSize(16777215, 16777215))
        self.val_satelits_count.setStyleSheet(u"border: none")

        self.status_bar.addWidget(self.val_satelits_count)

        self.val_dos_bat = QLabel(self.botom_status_bar)
        self.val_dos_bat.setObjectName(u"val_dos_bat")
        self.val_dos_bat.setMinimumSize(QSize(90, 0))
        self.val_dos_bat.setMaximumSize(QSize(16777215, 16777215))
        self.val_dos_bat.setStyleSheet(u"border: none")

        self.status_bar.addWidget(self.val_dos_bat)

        self.label_9 = QLabel(self.botom_status_bar)
        self.label_9.setObjectName(u"label_9")

        self.status_bar.addWidget(self.label_9)

        self.status_bar.setStretch(7, 1)

        self.verticalLayout.addWidget(self.botom_status_bar)

        self.verticalLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0431\u0440\u0430\u0442\u044c \u043f\u043e\u0440\u0442:", None))
        self.btn_conect.setText(QCoreApplication.translate("MainWindow", u"\u041f\u043e\u0434\u043b\u043a\u044e\u0447\u0438\u0442\u044c\u0441\u044f", None))
        self.val_dose_rate.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043e\u0437\u0430:", None))
        self.val_alt.setText(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0441\u043e\u0442\u0430:", None))
        self.val_gps_alt.setText(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0441\u043e\u0442\u0430 \u043f\u043e GPS:", None))
        self.val_sea_alt.setText(QCoreApplication.translate("MainWindow", u"\u0410\u0431\u0441\u043e\u043b\u044e\u0442\u043d\u0430\u044f \u0432\u044b\u0441\u043e\u0442\u0430:", None))
        self.val_calc_alt.setText(QCoreApplication.translate("MainWindow", u"\u041f\u0435\u0440\u0435\u0441\u0447\u0438\u0442\u0430\u043d\u0430\u044f \u0432\u044b\u0441\u043e\u0442\u0430:", None))
        self.val_sea_temp.setText(QCoreApplication.translate("MainWindow", u"\u0422\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430:", None))
        self.val_sea_hum.setText(QCoreApplication.translate("MainWindow", u"\u0412\u043b\u0430\u0436\u043d\u043e\u0441\u0442\u044c:", None))
        self.val_speed.setText(QCoreApplication.translate("MainWindow", u"\u0421\u043a\u043e\u0440\u043e\u0441\u0442\u044c:", None))
        self.val_volt.setText(QCoreApplication.translate("MainWindow", u"\u041d\u0430\u043f\u0440\u044f\u0436\u0435\u043d\u0438\u0435:", None))
        self.val_cur.setText(QCoreApplication.translate("MainWindow", u"\u0422\u043e\u043a:", None))
        self.val_lat.setText(QCoreApplication.translate("MainWindow", u"\u0428\u0438\u0440\u043e\u0442\u0430:", None))
        self.val_lon.setText(QCoreApplication.translate("MainWindow", u"\u0414\u043e\u043b\u0433\u043e\u0442\u0430:", None))
        self.val_date.setText(QCoreApplication.translate("MainWindow", u"01.01.2025", None))
        self.val_time.setText(QCoreApplication.translate("MainWindow", u"00:00:00", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"\u0412\u043e\u0441\u0442\u0430\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435 \u0434\u0430\u043d\u043d\u044b\u0445", None))
        self.nn_switcher.setText("")
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u0421\u0442\u0430\u0442\u0443\u0441 \u043f\u043e\u0434\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u044f: ", None))
        self.val_status_conect.setText("")
        self.val_msg_count.setText(QCoreApplication.translate("MainWindow", u"    \u0421\u043e\u043e\u0431\u0449\u0435\u0439\u043d\u0438\u0439:", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"    \u0421\u0442\u0430\u0442\u0443\u0441 GPS: ", None))
        self.val_status_gps.setText("")
        self.val_satelits_count.setText(QCoreApplication.translate("MainWindow", u"    C\u043f\u0443\u0442\u043d\u0438\u043a\u043e\u0432 GPS:", None))
        self.val_dos_bat.setText(QCoreApplication.translate("MainWindow", u"    \u0417\u0430\u0440\u044f\u0434 \u0434\u043e\u0437\u0438\u043c\u0435\u0442\u0440\u0430:", None))
        self.label_9.setText("")
    # retranslateUi

