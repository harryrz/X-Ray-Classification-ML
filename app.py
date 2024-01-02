import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

#some routing for displaying the home page
@app.route('/')
@app.route('/home')
def plot_graph():
  return render_template('plt_tmpl.html', name = "Model Performance", url1 ='static/images/my_plot1.png')