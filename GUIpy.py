from calendar import month
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QComboBox, QMessageBox
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore
import pickle
import numpy as np
import random as r

class App(QDialog):
    def __init__(self):
        super().__init__()
        self.title = "Smart Meter Prediction"
        self.setWindowIcon(QIcon('icons8-meter-64.png'))
        font = QFont()
        font.setFamily("Calibri")
        font.setPointSizeF(15.0)
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        windowLayout = QVBoxLayout()
        self.horizontalGroupBox1 = QGroupBox("  Please Enter Date, Month, and Year")
        self.horizontalGroupBox1.setStyleSheet("text-align:center;height:25px; background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgb(36, 196, 220), stop:1 rgb(81, 74, 157)); border-radius:5px; margin-top: 10px;")
        self.layout1 = QGridLayout()
        self.layout1.setColumnStretch(0, 2)
        self.layout1.setColumnStretch(1, 4)

        date = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
        month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        year = ['2012', '2013', '2014','2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032']
        time = ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00']
        
        self.QLDate = QLabel("Date")
        self.QLDate.setAlignment(QtCore.Qt.AlignCenter)
        self.QLDate.setStyleSheet("background-color:darkblue;color:white; font:bold;")
        self.qDate = QComboBox()
        self.qDate.setEditable(True)
        self.qDate.setStyleSheet("background-color:white;")
        self.qDate.addItems(date)
        self.qDate.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.layout1.addWidget(self.QLDate, 1, 0)
        self.layout1.addWidget(self.qDate, 1, 1)
        
        self.QLMonth = QLabel("Month")
        self.QLMonth.setAlignment(QtCore.Qt.AlignCenter)
        self.QLMonth.setStyleSheet("background-color:darkblue;color:white; font:bold;")
        self.qMonth = QComboBox()
        self.qMonth.setEditable(True)
        self.qMonth.setStyleSheet("background-color:white;")
        self.qMonth.addItems(month)
        self.qMonth.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.layout1.addWidget(self.QLMonth, 2, 0)
        self.layout1.addWidget(self.qMonth, 2, 1)

        self.QLYear = QLabel("Year")
        self.QLYear.setAlignment(QtCore.Qt.AlignCenter)
        self.QLYear.setStyleSheet("background-color:darkblue;color:white; font:bold;")
        self.qYear = QComboBox()
        self.qYear.setEditable(True)
        self.qYear.setStyleSheet("background-color:white;")
        self.qYear.addItems(year)
        self.qYear.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.layout1.addWidget(self.QLYear, 3, 0)
        self.layout1.addWidget(self.qYear, 3, 1)
        self.horizontalGroupBox1.setLayout(self.layout1)

        '''---------------------------------------------------------------------------------'''
        self.horizontalGroupBox2 = QGroupBox("  Please Enter Hour for Prediction")
        self.horizontalGroupBox2.setStyleSheet("height: 25px; background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgb(250, 250, 250), stop:1 rgb(205, 205, 205)); border-radius:5px; margin-top: 10px;")
        self.layout2 = QGridLayout()
        self.layout2.setColumnStretch(0, 2)
        self.layout2.setColumnStretch(1, 4)
        self.QLHour = QLabel("Hour")
        self.QLHour.setAlignment(QtCore.Qt.AlignCenter)
        self.QLHour.setStyleSheet("background-color:black;color:white; font:bold;")
        self.qHour = QComboBox()
        self.qHour.setEditable(True)
        self.qHour.setStyleSheet("background-color:white;")
        self.qHour.addItems(time)
        self.qHour.lineEdit().setAlignment(QtCore.Qt.AlignCenter)
        self.layout2.addWidget(self.QLHour, 1, 0)
        self.layout2.addWidget(self.qHour, 1, 1)

        self.horizontalGroupBox2.setLayout(self.layout2)
        '''---------------------------------------------------------------------------------'''

        self.horizontalGroupBox3 = QGroupBox("  Press Submit Button for Prediction or Clear Button to Clear all value")
        self.horizontalGroupBox3.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgb(36, 196, 220), stop:1 rgb(81, 74, 157)); border-radius:10px; margin-top: 10px;")
        self.layout3 = QGridLayout()
        self.layout3.setColumnStretch(0, 4)
        self.layout3.setColumnStretch(1, 2)
        self.button_predict = QPushButton()
        self.button_predict.setStyleSheet("background-color:white; height: 30px;")
        self.button_predict.setText("Predict")
        self.button_predict.clicked.connect(lambda: self.button_predict_clicked(self.qDate.currentText(), self.qMonth.currentText(), self.qYear.currentText(), self.qHour.currentText()))

        # add clear button
        self.button_reset = QPushButton()
        self.button_reset.setStyleSheet("background-color:white; height: 30px;")
        self.button_reset.setText("Clear")
        self.button_reset.clicked.connect(lambda: self.button_reset_clicked())

        self.layout3.addWidget(self.button_predict, 0, 0)
        self.layout3.addWidget(self.button_reset, 0, 1)

        self.horizontalGroupBox3.setLayout(self.layout3)

        '''---------------------------------------------------------------------------------'''

        windowLayout.addWidget(self.horizontalGroupBox1)
        windowLayout.addWidget(self.horizontalGroupBox2)
        windowLayout.addWidget(self.horizontalGroupBox3)

        self.setLayout(windowLayout)
        self.show()

    def showDialog(self,aircon,fridge,fan,phonecharger,laptop,airfilter,lightbulba,lightbulbb,lightbulbc):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText("You have opened:\n {0} Air Condition,\n {1} Fridge,\n {2} Fan,\n {3} Phone Charger,\n {4} Laptop,\n {5} AirFilter \nand {6} Light Bulb 1\n{7} Light Bulb 2\n{8} Light Bulb 3".format(aircon, fridge, fan, phonecharger, laptop, airfilter, lightbulba, lightbulbb,lightbulbc))
        msgBox.setWindowTitle("Appliances Used")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()    

    def button_reset_clicked(self):
        #print("button clear is pressed.")
        self.qDate.setCurrentText("1")
        self.qMonth.setCurrentText("Jan")
        self.qYear.setCurrentText("2022")
        self.qHour.setCurrentText("00:00")

    def button_predict_clicked(self, date, month, year, hour):
        n_date = int(date)

        # convert month (string) to int
        if month == "Jan":
            n_month = 1
        elif month == "Feb":
            n_month = 2
        elif month == "Mar":
            n_month = 3
        elif month == "Apr":
            n_month = 4
        elif month == "May":
            n_month = 5
        elif month == "Jun":
            n_month = 6
        elif month == "Jul":
            n_month = 7
        elif month == "Aug":
            n_month = 8
        elif month == "Sep":
            n_month = 9
        elif month == "Oct":
            n_month = 10
        elif month == "Nov":
            n_month = 11
        elif month == "Dec":
            n_month = 12
        else:
            n_month = ""

        # split year string into list
        n = 2
        year_chunks = [year[i:i+n] for i in range(0, len(year), n)]
        n_year = year_chunks[1]

        # calculate season - classifying by months of the year based on "tmg.go.th/info/info.php?FileID=53"
        # summer - February - May - value = 1
        # rainy - Jun - Sep - value = 2
        # winter - Oct - Jan - value = 3
        n_season = 0
        if month == "Feb" or month == "Mar" or month == "Apr" or month == "May":
            n_season = 1 # summer
        elif month == "Jun" or month == "Jul" or month == "Aug" or month == "Sep":
            n_season = 2 # rainy
        else:
            n_season = 3 # winter

        n_hour = 0
        # convert hour into numerical
        if hour == "00:00":
            n_hour = 0
        elif hour == "01:00":
            n_hour = 1
        elif hour == "02:00":
            n_hour = 2
        elif hour == "03:00":
            n_hour = 3
        elif hour == "04:00":
            n_hour = 4
        elif hour == "05:00":
            n_hour = 5
        elif hour == "06:00":
            n_hour = 6
        elif hour == "07:00":
            n_hour = 7
        elif hour == "08:00":
            n_hour = 8
        elif hour == "09:00":
            n_hour = 9
        elif hour == "10:00":
            n_hour = 10
        elif hour == "11:00":
            n_hour = 11
        elif hour == "12:00":
            n_hour = 12
        elif hour == "13:00":
            n_hour = 13
        elif hour == "14:00":
            n_hour = 14
        elif hour == "15:00":
            n_hour = 15
        elif hour == "16:00":
            n_hour = 16
        elif hour == "17:00":
            n_hour = 17
        elif hour == "18:00":
            n_hour = 18
        elif hour == "19:00":
            n_hour = 19
        elif hour == "20:00":
            n_hour = 20
        elif hour == "21:00":
            n_hour = 21
        elif hour == "22:00":
            n_hour = 22
        elif hour == "23:00":
            n_hour = 23
        else:
            n_hour = 0

        n_watts = r.randint(213,4612)
        # all necessary inputs are n_date, n_month, n_year, n_season, n_hour, n_watts.
        print("value: {0}, {1}, {2}, {3}, {4}, {5}".format(n_date, n_month, n_year, n_season, n_hour, n_watts))
        
        # load pre-trained model to predict the input
        filename = "rdfr.sav"
        loaded_model = pickle.load(open(filename, 'rb'))
        predicted = loaded_model.predict([[n_date, n_month, n_year, n_season, n_hour, n_watts]])
        
        # round prediction value
        round_predicted = np.round(predicted)

        aircon = round_predicted[0][0]
        fridge = round_predicted[0][1]
        fan = round_predicted[0][2]
        phonecharger = round_predicted[0][3]
        laptop = round_predicted[0][4]
        airfilter = round_predicted[0][5]
        lightbulba = round_predicted[0][6]
        lightbulbb = round_predicted[0][7]
        lightbulbc = round_predicted[0][8]
        self.showDialog(aircon,fridge,fan,phonecharger,laptop,airfilter,lightbulba, lightbulbb, lightbulbc)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())