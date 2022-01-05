# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\UFU\Disciplinas Outros Cursos\Computação Evolucionária\Tarefas\Tarefa 2 - Algoritmo Genético Simples\ga_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import style

from backend.GA_plotting_handler import GALivePlotHandler2D

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("Genetic Algorithm")
        MainWindow.resize(1200, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(450, 300))
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.topMenu = QtWidgets.QFrame(self.centralwidget)
        self.topMenu.setMaximumSize(QtCore.QSize(16777215, 45))
        self.topMenu.setStyleSheet("background-color: rgb(234, 234, 234);")
        self.topMenu.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.topMenu.setFrameShadow(QtWidgets.QFrame.Raised)
        self.topMenu.setObjectName("topMenu")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.topMenu)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.optimizeButton = QtWidgets.QPushButton(self.topMenu)
        self.optimizeButton.setMinimumSize(QtCore.QSize(0, 0))
        self.optimizeButton.setObjectName("optimizeButton")
        self.horizontalLayout_2.addWidget(self.optimizeButton)
        self.equationLineEd = QtWidgets.QLineEdit(self.topMenu)
        self.equationLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.equationLineEd.setText("")
        self.equationLineEd.setObjectName("equationLineEd")
        self.horizontalLayout_2.addWidget(self.equationLineEd)
        self.verticalLayout.addWidget(self.topMenu)
        self.content = QtWidgets.QFrame(self.centralwidget)
        self.content.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.content.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.content.setFrameShadow(QtWidgets.QFrame.Raised)
        self.content.setObjectName("content")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.content)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.paramsMenu = QtWidgets.QFrame(self.content)
        self.paramsMenu.setMinimumSize(QtCore.QSize(0, 0))
        self.paramsMenu.setMaximumSize(QtCore.QSize(200, 16777215))
        self.paramsMenu.setStyleSheet("background-color: rgb(185, 185, 185);")
        self.paramsMenu.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.paramsMenu.setFrameShadow(QtWidgets.QFrame.Raised)
        self.paramsMenu.setObjectName("paramsMenu")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.paramsMenu)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.paramsFrame = QtWidgets.QFrame(self.paramsMenu)
        self.paramsFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.paramsFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.paramsFrame.setObjectName("paramsFrame")
        self.gridLayout = QtWidgets.QGridLayout(self.paramsFrame)
        self.gridLayout.setObjectName("gridLayout")
        self.numGenLabel = QtWidgets.QLabel(self.paramsFrame)
        self.numGenLabel.setObjectName("numGenLabel")
        self.gridLayout.addWidget(self.numGenLabel, 0, 0, 1, 1)
        self.numGenLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.numGenLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.numGenLineEd.setObjectName("numGenLineEd")
        self.gridLayout.addWidget(self.numGenLineEd, 0, 1, 1, 1)
        self.mutRateLabel = QtWidgets.QLabel(self.paramsFrame)
        self.mutRateLabel.setObjectName("mutRateLabel")
        self.gridLayout.addWidget(self.mutRateLabel, 3, 0, 1, 1)
        self.mutRateLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.mutRateLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.mutRateLineEd.setObjectName("mutRateLineEd")
        self.gridLayout.addWidget(self.mutRateLineEd, 3, 1, 1, 1)
        self.popSizeLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.popSizeLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.popSizeLineEd.setObjectName("popSizeLineEd")
        self.gridLayout.addWidget(self.popSizeLineEd, 1, 1, 1, 1)
        self.crossRateLabel = QtWidgets.QLabel(self.paramsFrame)
        self.crossRateLabel.setObjectName("crossRateLabel")
        self.gridLayout.addWidget(self.crossRateLabel, 2, 0, 1, 1)
        self.popSizeLabel = QtWidgets.QLabel(self.paramsFrame)
        self.popSizeLabel.setObjectName("popSizeLabel")
        self.gridLayout.addWidget(self.popSizeLabel, 1, 0, 1, 1)
        self.crossRateLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.crossRateLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.crossRateLineEd.setObjectName("crossRateLineEd")
        self.gridLayout.addWidget(self.crossRateLineEd, 2, 1, 1, 1)
        self.genSizeLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.genSizeLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.genSizeLineEd.setObjectName("genSizeLineEd")
        self.gridLayout.addWidget(self.genSizeLineEd, 4, 1, 1, 1)
        self.genSizeLabel = QtWidgets.QLabel(self.paramsFrame)
        self.genSizeLabel.setObjectName("genSizeLabel")
        self.gridLayout.addWidget(self.genSizeLabel, 4, 0, 1, 1)
        self.intervalxLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.intervalxLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.intervalxLineEd.setObjectName("intervalxLineEd")
        self.gridLayout.addWidget(self.intervalxLineEd, 5, 1, 1, 1)
        self.intervalxLabel = QtWidgets.QLabel(self.paramsFrame)
        self.intervalxLabel.setObjectName("intervalxLabel")
        self.gridLayout.addWidget(self.intervalxLabel, 5, 0, 1, 1)
        self.intervalyLabel = QtWidgets.QLabel(self.paramsFrame)
        self.intervalyLabel.setObjectName("intervalyLabel")
        self.gridLayout.addWidget(self.intervalyLabel, 6, 0, 1, 1)
        self.intervalyLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.intervalyLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.intervalyLineEd.setObjectName("intervalyLineEd")
        self.gridLayout.addWidget(self.intervalyLineEd, 6, 1, 1, 1)

        self.elitismNumLabel = QtWidgets.QLabel(self.paramsFrame)
        self.elitismNumLabel.setObjectName("elitismNumLabel")
        self.gridLayout.addWidget(self.elitismNumLabel, 7, 0, 1, 1)
        self.elitismNumLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.elitismNumLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.elitismNumLineEd.setObjectName("elitismNumLineEd")
        self.gridLayout.addWidget(self.elitismNumLineEd, 7, 1, 1, 1)

        self.tournamentKLabel = QtWidgets.QLabel(self.paramsFrame)
        self.elitismNumLabel.setObjectName("tournamentKLabel")
        self.gridLayout.addWidget(self.tournamentKLabel, 8, 0, 1, 1)
        self.tournamentKLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.tournamentKLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.tournamentKLineEd.setObjectName("tournamentKLineEd")
        self.gridLayout.addWidget(self.tournamentKLineEd, 8, 1, 1, 1)

        self.selectionMethodLabel = QtWidgets.QLabel(self.paramsFrame)
        self.elitismNumLabel.setObjectName("selectionMethodLabel")
        self.gridLayout.addWidget(self.selectionMethodLabel, 9, 0, 1, 1)
        self.selectionMethodLineEd = QtWidgets.QLineEdit(self.paramsFrame)
        self.selectionMethodLineEd.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.selectionMethodLineEd.setObjectName("selectionMethodLineEd")
        self.gridLayout.addWidget(self.selectionMethodLineEd, 9, 1, 1, 1)

        self.verticalLayout_2.addWidget(self.paramsFrame)
        self.supportFrame = QtWidgets.QFrame(self.paramsMenu)
        self.supportFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.supportFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.supportFrame.setObjectName("supportFrame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.supportFrame)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.resultDisplayFrame = QtWidgets.QFrame(self.supportFrame)
        self.resultDisplayFrame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.resultDisplayFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.resultDisplayFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.resultDisplayFrame.setObjectName("resultDisplayFrame")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.resultDisplayFrame)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.resultDisplayFrameTop = QtWidgets.QFrame(self.resultDisplayFrame)
        self.resultDisplayFrameTop.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.resultDisplayFrameTop.setFrameShadow(QtWidgets.QFrame.Raised)
        self.resultDisplayFrameTop.setObjectName("resultDisplayFrameTop")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.resultDisplayFrameTop)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.bestFitLabel = QtWidgets.QLabel(self.resultDisplayFrameTop)
        self.bestFitLabel.setObjectName("bestFitLabel")
        self.gridLayout_2.addWidget(self.bestFitLabel, 0, 0, 1, 1)
        self.bestFitDisplay = QtWidgets.QLabel(self.resultDisplayFrameTop)
        self.bestFitDisplay.setText("")
        self.bestFitDisplay.setObjectName("bestFitDisplay")
        self.gridLayout_2.addWidget(self.bestFitDisplay, 0, 1, 1, 1)
        self.bestPointLabel = QtWidgets.QLabel(self.resultDisplayFrameTop)
        self.bestPointLabel.setObjectName("bestPointLabel")
        self.gridLayout_2.addWidget(self.bestPointLabel, 1, 0, 1, 1)
        self.bestPointDisplay = QtWidgets.QLabel(self.resultDisplayFrameTop)
        self.bestPointDisplay.setText("")
        self.bestPointDisplay.setObjectName("bestPointDisplay")
        self.gridLayout_2.addWidget(self.bestPointDisplay, 1, 1, 1, 1)
        self.verticalLayout_4.addWidget(self.resultDisplayFrameTop)
        self.resultDisplayFrameBott = QtWidgets.QFrame(self.resultDisplayFrame)
        self.resultDisplayFrameBott.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.resultDisplayFrameBott.setFrameShadow(QtWidgets.QFrame.Raised)
        self.resultDisplayFrameBott.setObjectName("resultDisplayFrameBott")
        self.verticalLayout_4.addWidget(self.resultDisplayFrameBott)
        self.verticalLayout_3.addWidget(self.resultDisplayFrame)
        self.verticalLayout_2.addWidget(self.supportFrame)

        self.verticalLayout_2.addWidget(self.supportFrame)
        self.horizontalLayout.addWidget(self.paramsMenu)
        self.plotFrame = QtWidgets.QFrame(self.content)
        self.plotFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plotFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.plotFrame.setObjectName("plotFrame")
        self.plotFrameLayout = QtWidgets.QVBoxLayout(self.plotFrame)
        self.plotFrameLayout.setContentsMargins(0, 0, 0, 0)
        self.plotFrameLayout.setSpacing(6)
        self.plotFrameLayout.setObjectName("plotFrameLayout")
        self.horizontalLayout.addWidget(self.plotFrame)
        self.verticalLayout.addWidget(self.content)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 857, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.setupPlotting()

        self.optimizeButton.clicked.connect(self.optimizeButtonClicked)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Genetic Algorithm"))
        self.optimizeButton.setText(_translate("MainWindow", "Optimize"))
        self.numGenLabel.setText(_translate("MainWindow", "Num generations:"))
        self.mutRateLabel.setText(_translate("MainWindow", "Mutation rate:"))
        self.crossRateLabel.setText(_translate("MainWindow", "Crossover rate:"))
        self.popSizeLabel.setText(_translate("MainWindow", "Population size:"))
        self.genSizeLabel.setText(_translate("MainWindow", "Genome size:"))
        self.intervalxLabel.setText(_translate("MainWindow", "Interval x:"))
        self.intervalyLabel.setText(_translate("MainWindow", "Interval y:"))
        self.bestFitLabel.setText(_translate("MainWindow", "Best fitness:"))
        self.bestPointLabel.setText(_translate("MainWindow", "Best point:"))
        self.elitismNumLabel.setText(_translate("MainWindow", "Elitism number:"))
        self.tournamentKLabel.setText(_translate("MainWindow", "Tournament K:"))
        self.selectionMethodLabel.setText(_translate("MainWindow", "Selection method:"))

        self.numGenLineEd.setText(_translate("MainWindow", "50"))
        self.mutRateLineEd.setText(_translate("MainWindow", "0.01"))
        self.crossRateLineEd.setText(_translate("MainWindow", "0.6"))
        self.popSizeLineEd.setText(_translate("MainWindow", "50"))
        self.genSizeLineEd.setText(_translate("MainWindow", "10"))
        self.intervalxLineEd.setText(_translate("MainWindow", "[0, 4]"))
        self.intervalyLineEd.setText(_translate("MainWindow", "[0, 2]"))
        self.elitismNumLineEd.setText(_translate("MainWindow", "1"))
        self.tournamentKLineEd.setText(_translate("MainWindow", "2"))
        self.selectionMethodLineEd.setText(_translate("MainWindow", "roulette"))
        self.equationLineEd.setText(_translate("MainWindow", "10 + x * np.sin(4 * x) + 3 * np.sin(2 * y)"))

    def setupPlotting(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self.plotFrame)

        self.plotFrameLayout.addWidget(self.canvas)
        self.plotFrameLayout.addWidget(self.toolbar)

    def optimizeButtonClicked(self):
        ga_parameters = {
            'pop_size': int(self.popSizeLineEd.text()),
            'num_generations': int(self.numGenLineEd.text()),
            'crossover_rate': float(self.crossRateLineEd.text()),
            'mutation_rate': float(self.mutRateLineEd.text()),
            'chromossome_size': int(self.genSizeLineEd.text()),
            'elitism_number': int(self.elitismNumLineEd.text()),
            'tournament_k': int(self.tournamentKLineEd.text()),
            'selection_method': self.selectionMethodLineEd.text(),
            }

        equation_string = self.equationLineEd.text()
        self.GA_live_plot_handler = GALivePlotHandler2D(
            equation_string=equation_string,
            intervals=[eval(self.intervalxLineEd.text()), eval(self.intervalyLineEd.text())],
            parameters=ga_parameters,
            )

        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(lambda: self.updateGUI()
            )
        self.timer.start()

    def updateGUI(self):
        self.GA_live_plot_handler.plot_GA(figure=self.figure, canvas=self.canvas, timer=self.timer)

        best_x, best_y, best_z = self.GA_live_plot_handler.get_best_xyz_values()
        self.bestFitDisplay.setText(str(round(best_z, 2)))
        self.bestPointDisplay.setText(str([round(best_x, 2), round(best_y, 2)]))


def init_gui():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setWindowIcon(QtGui.QIcon('frontend/logo.png'))
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())