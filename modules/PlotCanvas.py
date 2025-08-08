import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
import matplotlib.collections
plt.rcParams.update({
    "font.size": 12,  
    "axes.titlesize": 14,  
    "axes.labelsize": 12,  
    "xtick.labelsize": 10,  
    "ytick.labelsize": 10,  
    "legend.fontsize": 10,  
    "figure.figsize": (6, 4),  
    "figure.dpi": 100  
})
import numpy as np
from PyQt5.QtWidgets import (QApplication, QFileDialog, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "1"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"]             = "1"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class PlotCanvas(FigureCanvasQTAgg):
    plot_added = pyqtSignal(str, str)  
    plot_removed = pyqtSignal(str, str)
    def __init__(self, parent=None):
        self.aspect_ratio = 14 / 5
        self.fig = plt.figure(figsize=(8, 3))
        
        self.fig.patch.set_facecolor((220/255, 220/255, 220/255))        
        self.ax1 = self.fig.add_subplot(1, 2, 1)  
        self.ax2 = self.fig.add_subplot(1, 2, 2)
        
        super().__init__(self.fig)
        #self.setParent(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, False)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.x_data1, self.y_data1 = [], []
        self.x_data2, self.y_data2 = [], []
        
        self.ax1.set_title('Rheology', fontsize=16)
        self.ax2.set_title('MWD', fontsize=16)
        
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
                
        self.ax2.set_xscale('log')
        
        self.ax1.set_xlabel(r'$\omega$ (rad/s)', fontsize=14)
        self.ax1.set_ylabel("G', G'' (Pa)", fontsize=14)
        
        self.ax2.set_xlabel('M (g/mol)', fontsize=14)
        self.ax2.set_ylabel('dW/dlogM', fontsize=14)
        
        self.plots_ax1 = {}
        self.plots_ax2 = {}
        
        self.first_resize = False
        self.draw()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.first_resize == False:
            self.fig.tight_layout()
            self.draw_idle()
        self.first_resize = True
        
    def clear_plot1(self):
        for ax in [self.ax1]:  
            title = ax.get_title()
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            xscale = ax.get_xscale()
            yscale = ax.get_yscale()
            legend = ax.get_legend()  
    
            for label in list(self.plots_ax1.keys()):
                if isinstance(self.plots_ax1[label], (plt.Line2D, matplotlib.collections.PathCollection)):
                    self.plots_ax1.pop(label).remove()
                    self.plot_removed.emit(label, "ax1")
                    
                
            for text in ax.texts:
                text.remove()
                
            if legend is not None:
                legend.remove()
            
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            self.draw()
            
            
    def clear_plot2(self):
        for ax in [self.ax2]:  
            title = ax.get_title()
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            xscale = ax.get_xscale()
            yscale = ax.get_yscale()
            legend = ax.get_legend()  
            
            for label in list(self.plots_ax2.keys()):
                items = self.plots_ax2[label]
                if not isinstance(items, list):
                    items = [items]
    
                
                for item in items[:]:  
                    if isinstance(item, (plt.Line2D, matplotlib.collections.PathCollection)):
                        item.remove()
                        items.remove(item)
                        self.plot_removed.emit(label, "ax2")  
                
                del self.plots_ax2[label]
                        
                    
            for text in ax.texts:
                text.remove()
                
            if legend is not None:
                legend.remove()
            
            ax.set_title(title, fontsize=16)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)                            
            
            self.draw()
            
    def remove_single_plot(self, label):
        """
        Removes the 'Predicted' MWD plot from axis 2 (ax2).
        """
        if label in self.plots_ax2:
            items = self.plots_ax2[label]
            if not isinstance(items, list):
                items = [items]

            for item in items:
                if isinstance(item, (plt.Line2D, matplotlib.collections.PathCollection)):
                    item.remove()

            del self.plots_ax2[label]

            self.plot_removed.emit(label, "ax2")

            self.update_legend(self.ax2)
            self.draw()
        
    def autoscale_plot1(self):
        self.ax1.set_autoscale_on(True)
        line_plots = [line for line in self.ax1.get_lines() if line.get_visible()]
        scatter_plots = [scatter for scatter in self.ax1.collections if scatter.get_visible()] 
        
        if not line_plots and not scatter_plots:
            return
    
        scatter_xlim = [float('1e+20'), float('1e-20')]
        scatter_ylim = [float('1e+20'), float('1e-20')]
    
        for scatter in self.ax1.collections:  
            if scatter.get_visible():
                x_data, y_data = scatter.get_offsets().T 
                if len(x_data) > 0:  
                    scatter_xlim = [min(scatter_xlim[0], np.min(x_data))/2, max(scatter_xlim[1], np.max(x_data))*2]
                if len(y_data) > 0:  
                    scatter_ylim = [min(scatter_ylim[0], np.min(y_data))/2, max(scatter_ylim[1], np.max(y_data))*2]
    
        if line_plots:  
            self.ax1.relim()  
            self.ax1.autoscale_view() 
    
            line_xlim = self.ax1.get_xlim()
            line_ylim = self.ax1.get_ylim()
    
            combined_xlim = [min(line_xlim[0], scatter_xlim[0]), max(line_xlim[1], scatter_xlim[1])]
            combined_ylim = [min(line_ylim[0], scatter_ylim[0]), max(line_ylim[1], scatter_ylim[1])]
        else:
            combined_xlim = scatter_xlim
            combined_ylim = scatter_ylim
        
        self.ax1.set_xlim(combined_xlim)
        self.ax1.set_ylim(combined_ylim)
    
        self.fig.tight_layout()
        self.draw()
        QApplication.processEvents()
        
    def autoscale_plot2(self):
        self.ax2.set_autoscale_on(True)
        self.ax2.relim()  
        self.ax2.autoscale_view()
        self.fig.tight_layout()
        self.draw()
        QApplication.processEvents()
    
    def plot_scatter_on_axes1(self, x_data, y_data, facecolors, edgecolors, linewidth, label, size):

        scatter_obj = self.ax1.scatter(x_data, y_data, facecolors=facecolors, edgecolors=edgecolors, linewidth = linewidth, label = label, s = size)
        
        self.ax1.legend()
        
        self.draw()
        
        self.plots_ax1[label] = scatter_obj
        self.plot_added.emit(label, "ax1")
        
    def plot_line_on_axes1(self, x_data, y_data, linetype, color, label, linewidth):
        
        line_obj, = self.ax1.plot(x_data, y_data, linetype, color=color, label=label, linewidth=linewidth)
        self.ax1.legend()
        self.draw()
        
        self.plots_ax1[label] = line_obj
        self.plot_added.emit(label, "ax1")
        
    def plot_scatter_on_axes2(self, x_data, y_data, facecolors, edgecolors, linewidth, label, size):
        
        scatter_obj = self.ax2.scatter(
            x_data, y_data, facecolors=facecolors, edgecolors=edgecolors, 
            linewidth=linewidth, label=label, s=size
        )
        self.ax2.legend()
        self.draw()

        if label is None and self.recent_label is not None:
            if isinstance(self.plots_ax2[self.recent_label], list):
                self.plots_ax2[self.recent_label].append(scatter_obj)
            else:
                self.plots_ax2[self.recent_label] = [self.plots_ax2[self.recent_label], scatter_obj]
        else:
            self.plots_ax2[label] = scatter_obj
            self.recent_label = label  
            self.plot_added.emit(label, "ax2")
        
    def plot_line_on_axes2(self, x_data, y_data, linetype, color, label, linewidth):
        
        line_obj, = self.ax2.plot(
            x_data, y_data, linetype, color=color, label=label, linewidth=linewidth
        )
        self.ax2.legend()
        self.draw()

        if label is None and self.recent_label is not None:
            if isinstance(self.plots_ax2[self.recent_label], list):
                self.plots_ax2[self.recent_label].append(line_obj)
            else:
                self.plots_ax2[self.recent_label] = [self.plots_ax2[self.recent_label], line_obj]
        else:
            self.plots_ax2[label] = line_obj
            self.recent_label = label 
            self.plot_added.emit(label, "ax2")
        
    def toggle_visibility(self, label, axis, visible):
        if axis == "ax1":
            plot_dict = self.plots_ax1  
        else:
            plot_dict = self.plots_ax2
        if label in plot_dict:
            items = plot_dict[label]
            if not isinstance(items, list):
                items = [items]
            for item in items:
                item.set_visible(visible)
            if axis == "ax1":    
                self.update_legend(self.ax1)
                self.autoscale_plot1()
            else:
                self.update_legend(self.ax2)
                self.autoscale_plot2()
            self.draw()
            
    def update_legend(self, ax):
        handles, labels = [], []
        
        for handle in ax.get_lines() + ax.collections:
            if handle.get_visible() and not handle.get_label().startswith('_'):
                handles.append(handle)
                labels.append(handle.get_label())
        
        try: 
            a = handles[0]
            ax.legend(handles, labels)
        except:
            ax.get_legend().remove()
        QApplication.processEvents()
    
    def text_plot1(self, text, color, fontsize):
        self.ax1.text(0.5, 0.5, text, va='center', ha='center', color=color, fontsize=fontsize, fontweight='bold',transform=self.ax1.transAxes)
        self.draw()
        QApplication.processEvents()
        
    def change_axes_plot1(self, xlabel, ylabel):
        self.ax1.set_xlabel(xlabel, fontsize=14)
        self.ax1.set_ylabel(ylabel, fontsize=14)
                
    def save_ax_figure(self, ax):
        filename, _ = QFileDialog.getSaveFileName(
            None,
            "Save Axis as Figure",
            "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
    
        if not filename:
            return  
    
        plot_dict = self.plots_ax1 if ax == self.ax1 else self.plots_ax2
        fig, new_ax = plt.subplots(figsize=(8, 5))  
    
        keys = list(plot_dict.keys())

        if keys and keys[0] == "Predicted" and len(keys) > 1:
            keys[0], keys[1] = keys[1], keys[0]
        
        for label in keys:
            items = plot_dict[label]
            
            if not isinstance(items, list):
                items = [items]
    
            for item in items:
                if not item.get_visible():
                    continue  
    
                if isinstance(item, plt.Line2D):  
                    label = item.get_label()
                    color = item.get_color()
                    linewidth = item.get_linewidth()
                    linetype = item.get_linestyle()
    
                        
                    if linewidth > 1:
                        new_linewidth = linewidth - 1
                    else:
                        new_linewidth = linewidth
                    new_ax.plot(*item.get_data(), linetype, color=color, label=label, linewidth=new_linewidth)
    
                elif isinstance(item, matplotlib.collections.PathCollection):  
                    offsets = item.get_offsets().T
                    face_colors = item.get_facecolor()
                    edge_colors = item.get_edgecolor()
                    sizes = item.get_sizes()
    
                    if len(face_colors) > 0:
                        face_color = face_colors[0]
                    else:
                        face_color = "none"
    
                    if len(edge_colors) > 0:
                        edge_color = edge_colors[0]
                    else:
                        edge_color = "black"
    
                    new_ax.scatter(offsets[0], offsets[1], 
                                   label=item.get_label(), 
                                   c=[face_color],  
                                   edgecolors=[edge_color], 
                                   s=sizes)
    
        new_ax.set_title(ax.get_title())
        new_ax.set_xlabel(ax.get_xlabel())
        new_ax.set_ylabel(ax.get_ylabel())
        new_ax.set_xscale(ax.get_xscale())
        new_ax.set_yscale(ax.get_yscale())
    
        if new_ax.has_data():
            new_ax.legend()
    
        fig.savefig(filename, dpi=300)
        plt.close(fig)
